// build.rs for the `facedetector` CDylib (Windows only)
// This script configures Windows-specific build settings for the dynamic library.

fn main() {
    // Indicate Windows-specific compilation
    println!("cargo:rustc-cfg=windows_build");

    // Emit a build-time warning to confirm the target platform
    println!("cargo:warning=Building facedetector CDylib for Windows");

    // Optional: rerun if the main library source changes
    println!("cargo:rerun-if-changed=src/lib.rs");
}
