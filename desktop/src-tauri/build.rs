use std::fs;
use std::path::Path;

fn copy_dir_all(src: &Path, dst: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        if ty.is_dir() {
            copy_dir_all(&entry.path(), &dst.join(entry.file_name()))?;
        } else {
            let filename = entry.file_name();
            let name_str = filename.to_string_lossy();
            if name_str.ends_with(".py") || name_str.ends_with(".json") {
                fs::copy(entry.path(), dst.join(entry.file_name()))?;
            }
        }
    }
    Ok(())
}

fn main() {
    println!("cargo:rerun-if-changed=../../yams");
    println!("cargo:rerun-if-changed=../../yass");

    let yams_src = Path::new("../../yams");
    let cache_engine_yams = Path::new("../.cache/engine/yams");
    if yams_src.is_dir() && Path::new("../.cache/engine").is_dir() {
        let _ = copy_dir_all(yams_src, cache_engine_yams);
    }

    let yass_src = Path::new("../../yass");
    let cache_engine_yass = Path::new("../.cache/engine/yass");
    if yass_src.is_dir() && Path::new("../.cache/engine").is_dir() {
        let _ = copy_dir_all(yass_src, cache_engine_yass);
    }

    #[cfg(feature = "desktop")]
    tauri_build::build();
}
