use std::cmp::Ordering;

use crate::{gl_utils::*, LinSolve, SimulationSize};
use fruid::Array2D;
use glow::{
    Context as GlContext, HasContext, NativeBuffer, NativeProgram, NativeTexture, PixelPackData,
};
use rand::distributions::Uniform;
use rand::prelude::*;

#[test]
fn test_lin_solve() {
    let a = 1.;
    let c = 8.;
    let border = fruid::Bounds::Positive;
    assert_eq!(LinSolve::TOTAL_STEPS, 20);

    // Create GPU solver
    let (gl, _win, _, _ctx) = create_sdl2_context();

    let size = SimulationSize::from_tiles(20, 20);

    let solver = LinSolve::new(&gl, size).expect("Failed to create solver");

    // Create random data
    let mut rng = SmallRng::seed_from_u64(133769420);

    let cpu_x0 = random_data(size, &mut rng);
    let mut cpu_x = random_data(size, &mut rng);
    let (w, h) = size.dims();
    let mut cpu_scratch = Array2D::new(w, h);

    // Create gpu resources
    let gpu_x0 = create_image(&gl, w as i32, h as i32, Some(cpu_x0.data())).unwrap();

    let gpu_x = create_image(&gl, w as i32, h as i32, Some(cpu_x.data())).unwrap();

    // Solve on GPU
    let result = solver
        .step(&gl, border.into(), gpu_x, gpu_x0, a, c)
        .expect("Solver failed");

    let mut gpu_dl_x = Array2D::new(w, h);
    download_image(&gl, result, &mut gpu_dl_x);

    // Solve on CPU
    fruid::lin_solve(border, &mut cpu_x, &cpu_x0, &mut cpu_scratch, a, c);

    // Compare results
    let diffs: Vec<f32> = gpu_dl_x
        .data()
        .iter()
        .zip(cpu_x.data())
        .map(|(g, c)| (g - c).abs())
        .collect();

    let max_diff = *diffs.iter().max_by(|a, b| cmp_f32(*a, *b)).unwrap();
    let avg_diff = diffs.iter().sum::<f32>() / diffs.len() as f32;

    assert!(max_diff < 1e-5, "Max diff was {}, average {}", max_diff, avg_diff);
}

fn random_data(size: SimulationSize, rng: impl Rng) -> Array2D {
    let (w, h) = size.dims();
    let data = Uniform::new(-1., 1.).sample_iter(rng).take(w * h).collect();
    Array2D::from_array(w, data)
}


fn cmp_f32(a: &f32, b: &f32) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
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

impl From<fruid::Bounds> for crate::Bounds {
    fn from(f: fruid::Bounds) -> Self {
        match f {
            fruid::Bounds::NegX => crate::Bounds::NegX,
            fruid::Bounds::NegY => crate::Bounds::NegY,
            fruid::Bounds::Positive => crate::Bounds::Positive,
        }
    }
}
