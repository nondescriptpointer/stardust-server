use crate::{
	core::vulkano_data::VULKANO_CONTEXT,
	wayland::{Client, WaylandError, WaylandResult, core::surface::Surface},
};
use parking_lot::Mutex;
use std::{
	os::fd::{AsFd as _, OwnedFd},
	sync::{
		Arc, OnceLock,
		atomic::{AtomicBool, Ordering},
	},
};
use timeline_syncobj::{render_node::DrmRenderNode, timeline_syncobj::TimelineSyncObj};
use tracing::{error, info, warn};
use vulkano::{
	VulkanObject,
	sync::semaphore::{
		ExternalSemaphoreHandleType, ExternalSemaphoreHandleTypes, ImportSemaphoreFdInfo,
		Semaphore, SemaphoreCreateInfo, SemaphoreImportFlags,
	},
};
use waynest::ObjectId;
use waynest_protocols::server::staging::linux_drm_syncobj_v1::{
	wp_linux_drm_syncobj_manager_v1::WpLinuxDrmSyncobjManagerV1,
	wp_linux_drm_syncobj_surface_v1::WpLinuxDrmSyncobjSurfaceV1,
	wp_linux_drm_syncobj_timeline_v1::WpLinuxDrmSyncobjTimelineV1,
};
use waynest_server::Client as _;
use wgpu_hal::vulkan::{SIGNAL_SEMAPHORES, WAIT_SEMAPHORES};

static DRM_RENDER_NODE: OnceLock<DrmRenderNode> = OnceLock::new();

/// Whether VK_KHR_external_semaphore_fd is actually usable on the Vulkan device.
/// This is probed once at first use. Ash substitutes panic stubs for unloaded function
/// pointers, so we need to verify the extension works before calling import_fd/export_fd.
static EXT_SEMAPHORE_FD_AVAILABLE: OnceLock<bool> = OnceLock::new();
static EXT_SEMAPHORE_FD_WARNED: AtomicBool = AtomicBool::new(false);

fn is_ext_semaphore_fd_available() -> bool {
	*EXT_SEMAPHORE_FD_AVAILABLE.get_or_init(|| {
		let Some(vk) = VULKANO_CONTEXT.get() else {
			return false;
		};

		// Check if the physical device supports the extension at all
		if !vk.phys_dev.supported_extensions().khr_external_semaphore_fd {
			warn!("VK_KHR_external_semaphore_fd not supported by physical device; explicit sync disabled");
			return false;
		}

		// Directly check whether the Vulkan function pointers are loaded on the device.
		// The Vulkan device may have been created by wgpu/OpenXR without these extensions
		// actually enabled, even though vk_device_exts() told Vulkano they were. Ash
		// substitutes panic stubs (extern "system" fns) for unloaded function pointers,
		// and panicking across FFI is UB / process abort, so catch_unwind cannot help.
		// Instead, query vkGetDeviceProcAddr directly — it returns NULL for unloaded fns.
		let get_proc = vk.instance.fns().v1_0.get_device_proc_addr;
		let dev_handle = vk.dev.handle();
		let import_fn = unsafe {
			get_proc(
				dev_handle,
				b"vkImportSemaphoreFdKHR\0".as_ptr() as *const std::ffi::c_char,
			)
		};
		let export_fn = unsafe {
			get_proc(
				dev_handle,
				b"vkGetSemaphoreFdKHR\0".as_ptr() as *const std::ffi::c_char,
			)
		};

		if import_fn.is_none() || export_fn.is_none() {
			warn!(
				"VK_KHR_external_semaphore_fd function pointers not loaded \
				 (device likely created without the extension); \
				 falling back to blocking DRM syncobj sync"
			);
			return false;
		}

		info!("VK_KHR_external_semaphore_fd is available; explicit sync enabled");
		true
	})
}

fn get_drm_render_node() -> Option<&'static DrmRenderNode> {
	if let Some(node) = DRM_RENDER_NODE.get() {
		return Some(node);
	}
	let vk = VULKANO_CONTEXT.get()?;
	let render_node_id = vk.get_drm_render_node_id()?;
	let node = DrmRenderNode::new(render_node_id & 0xFF)
		.inspect_err(|err| error!("unable to open render_node: {err}"))
		.ok()?;
	_ = DRM_RENDER_NODE.set(node);
	DRM_RENDER_NODE.get()
}

// -- Timeline object --

#[derive(Debug, waynest_server::RequestDispatcher)]
#[waynest(error = crate::wayland::WaylandError, connection = crate::wayland::Client)]
pub struct SyncobjTimeline {
	pub id: ObjectId,
	pub syncobj: Arc<TimelineSyncObj>,
}

impl WpLinuxDrmSyncobjTimelineV1 for SyncobjTimeline {
	type Connection = Client;

	async fn destroy(&self, client: &mut Client, _sender_id: ObjectId) -> WaylandResult<()> {
		client.remove(self.id);
		Ok(())
	}
}

// -- Syncobj surface (per-surface explicit sync state) --

/// Pending sync point: a timeline + point to be committed with the next wl_surface.commit
#[derive(Debug, Clone)]
pub struct SyncPoint {
	pub timeline: Arc<TimelineSyncObj>,
	pub point: u64,
}

impl SyncPoint {
	/// Export a sync file from this timeline point and import it as a temporary Vulkan semaphore.
	pub fn to_vulkan_semaphore(&self) -> Option<Semaphore> {
		if !is_ext_semaphore_fd_available() {
			return None;
		}
		let vk = VULKANO_CONTEXT.get()?;
		let sema = Semaphore::from_pool(vk.dev.clone()).ok()?;
		let fd = self
			.timeline
			.export_sync_file_point(self.point)
			.inspect_err(|err| error!("failed to export sync file for acquire point: {err}"))
			.ok()?;
		unsafe {
			sema.import_fd(ImportSemaphoreFdInfo {
				flags: SemaphoreImportFlags::TEMPORARY,
				file: Some(fd.into()),
				..ImportSemaphoreFdInfo::handle_type(ExternalSemaphoreHandleType::SyncFd)
			})
			.inspect_err(|err| error!("failed to import sync fd into vulkan semaphore: {err}"))
			.ok()?;
		}
		Some(sema)
	}

	/// Signal this timeline point by importing a sync file from a Vulkan semaphore that has
	/// already been signaled (or has a signal operation queued).
	pub fn signal_from_semaphore(&self, semaphore: &Semaphore) {
		if !is_ext_semaphore_fd_available() {
			// Fall back to direct CPU signal
			self.signal_direct();
			return;
		}
		let fd = unsafe {
			semaphore
				.export_fd(ExternalSemaphoreHandleType::SyncFd)
				.inspect_err(|err| {
					error!("failed to export sync fd from release semaphore: {err}")
				})
		};
		if let Ok(fd) = fd {
			if let Err(err) = self.timeline.import_sync_file_point(fd.as_fd(), self.point) {
				error!("failed to import sync file into release timeline point: {err}");
			}
		}
	}

	/// CPU-side signal of this timeline point (fallback when no GPU semaphore is available).
	pub fn signal_direct(&self) {
		unsafe {
			if let Err(err) = self.timeline.signal(self.point) {
				error!("failed to directly signal release timeline point: {err}");
			}
		}
	}
}

/// The committed sync state for a surface (travels through the triple-buffer pipeline).
#[derive(Debug, Clone)]
pub struct SurfaceSyncState {
	pub acquire: SyncPoint,
	pub release: SyncPoint,
}

impl SurfaceSyncState {
	/// Push the acquire semaphore into WAIT_SEMAPHORES for the current frame.
	/// If Vulkan external semaphore support is not available, falls back to a
	/// blocking wait on the DRM timeline syncobj.
	pub fn push_acquire_semaphore(&self) {
		if let Some(sema) = self.acquire.to_vulkan_semaphore() {
			WAIT_SEMAPHORES.lock().push(sema.handle());
			// Store the semaphore so the handle stays valid until GPU consumes it.
			ACQUIRE_SEMAS.lock().push(sema);
		} else if !EXT_SEMAPHORE_FD_WARNED.swap(true, Ordering::Relaxed) {
			warn!(
				"Vulkan external semaphore not available; \
				 falling back to blocking DRM syncobj wait for acquire sync"
			);
		}
		// If we couldn't create a Vulkan semaphore, do a blocking wait
		// on the DRM timeline syncobj so the client's GPU work finishes
		// before the compositor reads.
		if !is_ext_semaphore_fd_available() {
			if let Err(err) = self.acquire.timeline.blocking_wait(self.acquire.point, None) {
				error!("failed blocking wait on acquire timeline point: {err}");
			}
		}
	}

	/// Create a Vulkan semaphore for release signaling, push it into SIGNAL_SEMAPHORES,
	/// and return a ReleaseSignaler that will import the sync file into the timeline on drop.
	pub fn create_release_signal(&self) -> Option<ReleaseSignaler> {
		if !is_ext_semaphore_fd_available() {
			return None;
		}
		let vk = VULKANO_CONTEXT.get()?;
		let sema = Semaphore::new(
			vk.dev.clone(),
			SemaphoreCreateInfo {
				export_handle_types: ExternalSemaphoreHandleTypes::SYNC_FD,
				..Default::default()
			},
		)
		.inspect_err(|err| error!("failed to create release semaphore: {err}"))
		.ok()?;
		let raw_sema = sema.handle();
		SIGNAL_SEMAPHORES.lock().push(raw_sema);
		Some(ReleaseSignaler {
			semaphore: sema,
			release_point: self.release.clone(),
		})
	}
}

/// Holds a release semaphore and its associated timeline point.
/// When the GPU has finished rendering (and signaled the semaphore),
/// call `signal()` to transfer the fence to the timeline.
#[derive(Debug)]
pub struct ReleaseSignaler {
	semaphore: Semaphore,
	release_point: SyncPoint,
}
impl ReleaseSignaler {
	/// Import the GPU-signaled semaphore's sync file into the release timeline point.
	#[allow(dead_code)]
	pub fn signal(self) {
		self.release_point.signal_from_semaphore(&self.semaphore);
	}
}
impl Drop for ReleaseSignaler {
	fn drop(&mut self) {
		// Import the GPU-signaled semaphore's sync file into the release timeline point.
		// signal_from_semaphore falls back to direct CPU signal if the extension is not available.
		self.release_point.signal_from_semaphore(&self.semaphore);
	}
}

// Storage for acquire semaphores that need to stay alive until GPU consumes them
static ACQUIRE_SEMAS: Mutex<Vec<Semaphore>> = Mutex::new(Vec::new());

/// Call this after rendering to clean up acquire semaphores from the current frame.
pub fn cleanup_acquire_semaphores() {
	ACQUIRE_SEMAS.lock().clear();
}

// -- Syncobj surface protocol object --

#[derive(Debug, waynest_server::RequestDispatcher)]
#[waynest(error = crate::wayland::WaylandError, connection = crate::wayland::Client)]
pub struct SyncobjSurface {
	pub id: ObjectId,
	pub surface_id: ObjectId,
	pending_acquire: Mutex<Option<SyncPoint>>,
	pending_release: Mutex<Option<SyncPoint>>,
}

impl SyncobjSurface {
	pub fn new(id: ObjectId, surface_id: ObjectId) -> Self {
		Self {
			id,
			surface_id,
			pending_acquire: Mutex::new(None),
			pending_release: Mutex::new(None),
		}
	}

	/// Take the pending sync state (called on wl_surface.commit).
	/// Returns Some if both acquire and release were set.
	pub fn take_pending(&self) -> Option<SurfaceSyncState> {
		let acquire = self.pending_acquire.lock().take()?;
		let release = self.pending_release.lock().take()?;
		Some(SurfaceSyncState { acquire, release })
	}
}

impl WpLinuxDrmSyncobjSurfaceV1 for SyncobjSurface {
	type Connection = Client;

	async fn destroy(&self, client: &mut Client, _sender_id: ObjectId) -> WaylandResult<()> {
		// Clear any pending state
		self.pending_acquire.lock().take();
		self.pending_release.lock().take();

		// Remove the syncobj surface reference from the wl_surface
		if let Some(surface) = client.get::<Surface>(self.surface_id) {
			surface.clear_syncobj_surface();
		}

		client.remove(self.id);
		Ok(())
	}

	async fn set_acquire_point(
		&self,
		client: &mut Client,
		_sender_id: ObjectId,
		timeline: ObjectId,
		point_hi: u32,
		point_lo: u32,
	) -> WaylandResult<()> {
		let timeline_obj = client
			.get::<SyncobjTimeline>(timeline)
			.ok_or(WaylandError::MissingObject(timeline))?;
		let point = ((point_hi as u64) << 32) | (point_lo as u64);
		*self.pending_acquire.lock() = Some(SyncPoint {
			timeline: timeline_obj.syncobj.clone(),
			point,
		});
		Ok(())
	}

	async fn set_release_point(
		&self,
		client: &mut Client,
		_sender_id: ObjectId,
		timeline: ObjectId,
		point_hi: u32,
		point_lo: u32,
	) -> WaylandResult<()> {
		let timeline_obj = client
			.get::<SyncobjTimeline>(timeline)
			.ok_or(WaylandError::MissingObject(timeline))?;
		let point = ((point_hi as u64) << 32) | (point_lo as u64);
		*self.pending_release.lock() = Some(SyncPoint {
			timeline: timeline_obj.syncobj.clone(),
			point,
		});
		Ok(())
	}
}

// -- Syncobj manager (global) --

#[derive(Debug, waynest_server::RequestDispatcher)]
#[waynest(error = crate::wayland::WaylandError, connection = crate::wayland::Client)]
pub struct SyncobjManager {
	pub id: ObjectId,
}

impl WpLinuxDrmSyncobjManagerV1 for SyncobjManager {
	type Connection = Client;

	async fn destroy(&self, client: &mut Client, _sender_id: ObjectId) -> WaylandResult<()> {
		client.remove(self.id);
		Ok(())
	}

	async fn get_surface(
		&self,
		client: &mut Client,
		_sender_id: ObjectId,
		id: ObjectId,
		surface: ObjectId,
	) -> WaylandResult<()> {
		let surface_obj = client
			.get::<Surface>(surface)
			.ok_or(WaylandError::MissingObject(surface))?;

		// Check if surface already has a syncobj surface associated
		if surface_obj.has_syncobj_surface() {
			return Err(WaylandError::Fatal {
				object_id: self.id,
				code: waynest_protocols::server::staging::linux_drm_syncobj_v1::wp_linux_drm_syncobj_manager_v1::Error::SurfaceExists as u32,
				message: "Surface already has a syncobj surface associated",
			});
		}

		let syncobj_surface = client.insert(id, SyncobjSurface::new(id, surface))?;
		surface_obj.set_syncobj_surface(syncobj_surface);

		info!("Created syncobj surface for wl_surface {}", surface);
		Ok(())
	}

	async fn import_timeline(
		&self,
		client: &mut Client,
		_sender_id: ObjectId,
		id: ObjectId,
		fd: OwnedFd,
	) -> WaylandResult<()> {
		let render_node = get_drm_render_node().ok_or(WaylandError::Fatal {
			object_id: self.id,
			code: waynest_protocols::server::staging::linux_drm_syncobj_v1::wp_linux_drm_syncobj_manager_v1::Error::InvalidTimeline as u32,
			message: "DRM render node not available",
		})?;

		let syncobj = TimelineSyncObj::import(render_node, fd.as_fd()).map_err(|err| {
			error!("Failed to import timeline syncobj: {err}");
			WaylandError::Fatal {
				object_id: self.id,
				code: waynest_protocols::server::staging::linux_drm_syncobj_v1::wp_linux_drm_syncobj_manager_v1::Error::InvalidTimeline as u32,
				message: "Failed to import timeline syncobj",
			}
		})?;

		client.insert(
			id,
			SyncobjTimeline {
				id,
				syncobj: Arc::new(syncobj),
			},
		)?;

		info!("Imported timeline syncobj as {}", id);
		Ok(())
	}
}
