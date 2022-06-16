use anyhow::{bail, format_err, Context as AnyhowContext, Result};
use glow::{Context as GlContext, HasContext, NativeBuffer, NativeProgram, NativeTexture, PixelPackData};
use std::path::Path;

/// Compile and link program from sources
pub fn create_program<P: AsRef<Path>>(
    gl: &GlContext,
    shader_sources: &[(u32, P)],
) -> Result<NativeProgram> {
    unsafe {
        let program = gl
            .create_program()
            .map_err(|e| format_err!("{:#}", e))
            .context("Cannot create program")?;

        let mut shaders = Vec::with_capacity(shader_sources.len());

        for (shader_type, shader_path) in shader_sources.iter() {
            // Read
            let shader_source = std::fs::read_to_string(shader_path)
                .with_context(|| format!("Failed to read {}", shader_path.as_ref().display()))?;

            // Compile
            let shader = gl
                .create_shader(*shader_type)
                .map_err(|e| format_err!("{:#}", e))
                .context("Cannot create program")?;

            gl.shader_source(shader, &shader_source);
            gl.compile_shader(shader);

            if !gl.get_shader_compile_status(shader) {
                bail!("{}", gl.get_shader_info_log(shader));
            }

            // Attach
            gl.attach_shader(program, shader);
            shaders.push(shader);
        }

        // Link
        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            bail!("{}", gl.get_program_info_log(program));
        }

        // Cleanup
        for shader in shaders {
            gl.detach_shader(program, shader);
            gl.delete_shader(shader);
        }

        Ok(program)
    }
}

/// Create a single-channel float image with the given dimensions 
pub fn create_image(
    gl: &GlContext,
    width: i32,
    height: i32,
    pixels: Option<&[f32]>,
) -> Result<NativeTexture> {
    unsafe {
        let tex = gl.create_texture()
            .map_err(|e| format_err!("{:#}", e))
            .context("Cannot create program")?;

        gl.bind_texture(glow::TEXTURE_2D, Some(tex));

        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_WRAP_S,
            glow::CLAMP_TO_EDGE as _,
            //glow::CLAMP_TO_BORDER as _,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_WRAP_T,
            glow::CLAMP_TO_EDGE as _,
            //glow::CLAMP_TO_BORDER as _,
        );

        //gl.tex_parameter_f32_slice(glow::TEXTURE_2D, glow::TEXTURE_BORDER_COLOR, &[0.; 4]);

        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MAG_FILTER,
            glow::LINEAR as _,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MIN_FILTER,
            glow::LINEAR as _,
        );

        gl.bind_image_texture(0, tex, 0, false, 0, glow::READ_WRITE, glow::R32F);

        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::R32F as _,
            width,
            height,
            0,
            glow::RED,
            glow::FLOAT,
            pixels.map(bytemuck::cast_slice),
        );

        Ok(tex)
    }
}

