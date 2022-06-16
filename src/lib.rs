use anyhow::{bail, format_err, Context as AnyhowContext, Result};
use glow::{
    Context as GlContext, HasContext, NativeBuffer, NativeProgram, NativeTexture, PixelPackData,
};
use std::path::Path;

mod gl_utils;
use gl_utils::*;

#[cfg(test)]
mod test;

/// Local size used in GPU kernels for X and for Y.
/// `LOCAL_SIZE * LOCAL_SIZE < MAX_COMPUTE_WORK_GROUP_INVOCATIONS`
const LOCAL_SIZE: usize = 32;

/// Boundary condition settings
#[repr(i32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Bounds {
    Positive = 0,
    NegX = 1,
    NegY = 2,
}

/// Simulation dimensions. Stored as tile counts, actual dimensions depend on solver kernels
#[derive(Copy, Clone, Debug)]
pub struct SimulationSize {
    pub x_tiles: usize,
    pub y_tiles: usize,
}

impl SimulationSize {
    /// Calculate the dimensions of the simulation volume (including boundaries)
    pub fn dims(&self) -> (usize, usize) {
        (
            self.x_tiles * LinSolve::TILE_OUTPUT_SIZE,
            self.y_tiles * LinSolve::TILE_OUTPUT_SIZE,
        )
    }
}

struct LinSolve {
    /// Scratch space used in-between dispatches
    sim_scratch: NativeTexture,
    /// Scratch space used during dispatch (large!)
    wg_scratch: NativeTexture,
    program: NativeProgram,
    size: SimulationSize,
}

impl LinSolve {
    // NOTE: These constants MUST match those specified in the shaders

    /// Number of steps each solver dispatch comprises
    const STEPS_PER_DISPATCH: usize = 5;

    /// Number of dispatches per solver step
    const N_DISPATCHES: usize = 4;

    /// Total steps per dispatch
    const TOTAL_STEPS: usize = Self::STEPS_PER_DISPATCH * Self::N_DISPATCHES;

    /// Each tile outputs information with the given width
    const TILE_OUTPUT_SIZE: usize = LOCAL_SIZE - Self::STEPS_PER_DISPATCH * 2;

    /// Create a new solver (also creates scratch space)
    pub fn new(gl: &GlContext, size: SimulationSize) -> Result<Self> {
        let program = create_program(gl, &[(glow::COMPUTE_SHADER, "./kernels/lin_solve.comp")])?;

        let workgroup_scratch = create_image(
            gl,
            (size.x_tiles * LOCAL_SIZE) as i32,
            (size.y_tiles * LOCAL_SIZE) as i32,
            None,
        )?;

        let (width, height) = size.dims();
        let sim_scratch = create_image(gl, width as i32, height as i32, None)?;

        Ok(Self {
            program,
            wg_scratch: workgroup_scratch,
            sim_scratch,
            size,
        })
    }

    /// Solve the given system, returning the texture containing the result.
    /// Return value may not be either of the two supplied textures!
    pub fn step(
        &self,
        gl: &GlContext,
        b: Bounds,
        x: NativeTexture,
        x0: NativeTexture,
        a: f32,
        c: f32,
    ) -> Result<NativeTexture> {
        const X0_BIND: u32 = 0;
        const READ_BIND: u32 = 1;
        const WRITE_BIND: u32 = 2;
        const SCRATCH_BIND: u32 = 3;

        unsafe {
            gl.bind_image_texture(X0_BIND, x0, 0, false, 0, glow::READ_WRITE, glow::R32F);
            gl.bind_image_texture(SCRATCH_BIND, self.wg_scratch, 0, false, 0, glow::READ_WRITE, glow::R32F);
        }

        let mut write_tex = x;
        let mut read_tex;
        for i in 0..Self::N_DISPATCHES {
            if i & 1 == 0 {
                read_tex = x;
                write_tex = self.sim_scratch;
            } else {
                read_tex = self.sim_scratch;
                write_tex = x;
            };

            unsafe {
                gl.bind_image_texture(READ_BIND, read_tex, 0, false, 0, glow::READ_WRITE, glow::R32F);
                gl.bind_image_texture(WRITE_BIND, write_tex, 0, false, 0, glow::READ_WRITE, glow::R32F);

                gl.dispatch_compute(self.size.x_tiles as u32, self.size.y_tiles as u32, 1);
                gl.memory_barrier(glow::SHADER_STORAGE_BARRIER_BIT);
            }
        }

        Ok(write_tex)
    }
}
