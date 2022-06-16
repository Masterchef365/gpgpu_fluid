use glow::{Context as GlContext, HasContext, NativeBuffer, NativeProgram, NativeTexture, PixelPackData};
use fruid::Array2D;
use crate::{gl_utils::*, LinSolve, SimulationSize};

#[test]
fn test_lin_solve() {
    let (gl, _win, _, _ctx) = create_sdl2_context();

    let size = SimulationSize::from_tiles(20, 20);

    let solver = LinSolve::new(&gl, size).expect("Failed to create solver");
}

/// Transfer image data from GPU to CPU
fn download_image(gl: &GlContext, src: NativeTexture, dest: &mut Array2D) {
    unsafe {
        gl.bind_texture(glow::TEXTURE_2D, Some(src));

        gl.get_tex_image(
            glow::TEXTURE_2D,
            0,
            glow::RED,
            glow::FLOAT,
            PixelPackData::Slice(bytemuck::cast_slice_mut(dest.data_mut())),
        );
    }
}

impl Into<fruid::Bounds> for crate::Bounds {
    fn into(self) -> fruid::Bounds {
        match self {
            crate::Bounds::NegX => fruid::Bounds::NegX,
            crate::Bounds::NegY => fruid::Bounds::NegY,
            crate::Bounds::Positive => fruid::Bounds::Positive,
        }
    }
}
