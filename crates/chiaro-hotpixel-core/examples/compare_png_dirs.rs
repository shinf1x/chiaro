//! Compare every PNG under two directories pixel by pixel.
//!
//! ```bash
//! cargo run -p chiaro-hotpixel-core --release --example compare_png_dirs -- before/ after/
//! ```
//!
//! Exits non-zero when any decoded image differs, so a refactor of the
//! pipeline can be proven output-identical on real captures.

use std::{env, fs::File, path::Path, process::exit};

use walkdir::WalkDir;

fn decode(path: &Path) -> Result<(u32, u32, png::ColorType, Vec<u8>), String> {
    let decoder = png::Decoder::new(File::open(path).map_err(|e| e.to_string())?);
    let mut reader = decoder.read_info().map_err(|e| e.to_string())?;
    let mut buffer = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buffer).map_err(|e| e.to_string())?;
    buffer.truncate(info.buffer_size());
    Ok((info.width, info.height, info.color_type, buffer))
}

fn main() {
    let mut args = env::args().skip(1);
    let (Some(left), Some(right)) = (args.next(), args.next()) else {
        eprintln!("usage: compare_png_dirs <dir-a> <dir-b>");
        exit(2);
    };
    let (left, right) = (Path::new(&left), Path::new(&right));
    let mut compared = 0usize;
    let mut different = 0usize;
    for entry in WalkDir::new(left).into_iter().filter_map(Result::ok) {
        if entry
            .path()
            .extension()
            .is_none_or(|e| !e.eq_ignore_ascii_case("png"))
        {
            continue;
        }
        let relative = entry.path().strip_prefix(left).unwrap();
        let other = right.join(relative);
        let a = decode(entry.path());
        let b = decode(&other);
        compared += 1;
        match (a, b) {
            (Ok(a), Ok(b)) if a == b => {}
            (Ok(a), Ok(b)) => {
                different += 1;
                let differing = a.3.iter().zip(&b.3).filter(|(x, y)| x != y).count();
                println!(
                    "DIFFERENT {}: {}x{} {:?} vs {}x{} {:?}, {differing} differing bytes",
                    relative.display(),
                    a.0,
                    a.1,
                    a.2,
                    b.0,
                    b.1,
                    b.2
                );
            }
            (Err(e), _) | (_, Err(e)) => {
                different += 1;
                println!("ERROR {}: {e}", relative.display());
            }
        }
    }
    println!("{compared} files compared, {different} differ");
    if different > 0 || compared == 0 {
        exit(1);
    }
}
