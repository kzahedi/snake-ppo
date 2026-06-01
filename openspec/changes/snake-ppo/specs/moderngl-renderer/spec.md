## ADDED Requirements

### Requirement: Offscreen headless rendering
The renderer SHALL support a headless mode using `moderngl.create_standalone_context()` that produces RGBA frame arrays without opening any window. Headless mode SHALL work during overnight training without a display server. Frame output SHALL be a numpy uint8 array of shape (height, width, 3).

#### Scenario: Offscreen frame capture
- **WHEN** `renderer.render_frame(state)` is called in headless mode
- **THEN** a numpy array of shape (H_px, W_px, 3) is returned with pixel values in [0, 255]

#### Scenario: No window appears in headless mode
- **WHEN** the renderer is constructed with `mode="offscreen"`
- **THEN** no OS window is created

### Requirement: Windowed mode with pygame backend
The renderer SHALL support a windowed mode using pygame as the moderngl window backend. The window SHALL display the game at a configurable pixel resolution (default 800×800) at up to 60 fps. The window SHALL remain responsive (processing pygame events) between frames.

#### Scenario: Window opens and displays content
- **WHEN** renderer is constructed with `mode="window"` and `renderer.show(state)` is called
- **THEN** a window appears and the game state is visible

#### Scenario: Window can be closed
- **WHEN** the user closes the window or presses Q
- **THEN** `renderer.should_quit()` returns True

### Requirement: Snake body gradient shader
The snake body SHALL be rendered as a series of quad segments. Each segment SHALL be coloured using a GLSL vertex shader that interpolates colour along the body from a bright head colour (configurable, default cyan #00FFCC) to a dark tail colour (configurable, default dark blue #003366) based on the segment's normalized index (0.0 = head, 1.0 = tail tip).

#### Scenario: Head segment is brightest
- **WHEN** a snake of length ≥ 3 is rendered
- **THEN** the pixel at the head position has higher luminance than the pixel at the tail position

#### Scenario: Gradient is smooth
- **WHEN** a snake of length 10 is rendered
- **THEN** adjacent segments differ in colour by a small, consistent amount (no abrupt jumps)

### Requirement: Food glow shader
Food SHALL be rendered as a point sprite with a radial glow effect implemented in a GLSL fragment shader. The glow SHALL pulse in intensity using a sinusoidal time uniform (period ≈ 1.5 seconds). The food colour SHALL be configurable (default warm orange #FF6B35).

#### Scenario: Food renders with glow
- **WHEN** a frame is rendered with food present
- **THEN** pixels surrounding the food cell have non-zero intensity that falls off radially

#### Scenario: Glow pulses over time
- **WHEN** frames are rendered at t=0 and t=0.75 seconds
- **THEN** the peak pixel intensity at the food location differs between the two frames

### Requirement: Value heatmap overlay
The renderer SHALL support an optional value heatmap overlay that colours each grid cell by its estimated value V(s). The overlay SHALL use a diverging colour map (cool blue for low values, neutral grey at zero, warm orange for high values). The overlay SHALL be blended over the game render at configurable opacity (default 0.4). The heatmap is disabled by default.

#### Scenario: Heatmap disabled by default
- **WHEN** renderer is constructed without specifying `heatmap=True`
- **THEN** rendered frames contain only game elements, no colour overlay

#### Scenario: Heatmap blends correctly
- **WHEN** renderer is constructed with `heatmap=True` and a value grid is provided
- **THEN** each grid cell's colour is a blend of its game colour and the heatmap colour at the specified opacity

### Requirement: Configurable resolution and cell size
The renderer SHALL derive cell pixel size from the output resolution and grid size (cell_px = resolution // grid_size). Both offscreen and windowed modes SHALL use the same resolution parameter (default 800×800).

#### Scenario: Cell size scales with grid
- **WHEN** grid_size=8 and resolution=800
- **THEN** each cell is rendered as a 100×100 pixel square

#### Scenario: Cell size scales with large grid
- **WHEN** grid_size=32 and resolution=800
- **THEN** each cell is rendered as a 25×25 pixel square
