pub mod area_lut;

fn main() {
    let area_lut_path = "textures/area-lut.png";
    area_lut::generate_area_lut(area_lut_path);
    println!("cargo:rerun-if-changed=build");
    println!("cargo:rerun-if-changed={area_lut_path}");
}
