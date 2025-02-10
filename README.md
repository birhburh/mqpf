# mqpf - port of pathfinder's essence to macroquad

Library to render pixel-perfect paths using macroquad

Currently this lib is mostly uses code of d3d9 implementation of pathfinder but reduced to a single file with support of only filled free-form shapes

### Attribution
- Heavily based on [pathfinder](https://github.com/servo/pathfinder)
- textures/area-lut.png generated using pathfinder's utils/area-lut package

### TODO:
- Make it work with webgl 1 and metal:
    - store mask inside rgba8 texture instead of rgba16f if possible
    - write metal shaders
- Reduce size so it will take around 1000 lines without docs