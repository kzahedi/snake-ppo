from __future__ import annotations

import numpy as np
import moderngl


# ---------------------------------------------------------------------------
# GLSL shaders
# ---------------------------------------------------------------------------

_BODY_VERT = """
#version 330
in vec2 in_pos;
in vec3 in_color;
out vec3 v_color;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_color = in_color;
}
"""

_BODY_FRAG = """
#version 330
in vec3 v_color;
out vec4 f_color;
void main() {
    f_color = vec4(v_color, 1.0);
}
"""

_FOOD_VERT = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
}
"""

_FOOD_FRAG = """
#version 330
in vec2 v_uv;
out vec4 f_color;
uniform vec3 u_color;
uniform float u_time;
void main() {
    vec2 d = v_uv - vec2(0.5);
    float dist = length(d) * 2.0;
    float pulse = 0.75 + 0.25 * sin(u_time * 4.189);
    float core  = 1.0 - smoothstep(0.0, 0.45, dist);
    float glow  = (1.0 - smoothstep(0.3, 1.0, dist)) * pulse * 0.5;
    float alpha = clamp(core + glow, 0.0, 1.0);
    vec3  col   = u_color + vec3(0.5, 0.35, 0.1) * glow;
    f_color = vec4(col * alpha, alpha);
}
"""

_HEAT_VERT = """
#version 330
in vec2  in_pos;
in float in_value;
out float v_value;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_value = in_value;
}
"""

_HEAT_FRAG = """
#version 330
in float v_value;
out vec4 f_color;
uniform float u_opacity;
void main() {
    vec3 cold = vec3(0.1, 0.3, 0.85);
    vec3 mid  = vec3(0.25, 0.25, 0.25);
    vec3 warm = vec3(0.9, 0.5, 0.1);
    float t = clamp(v_value, -1.0, 1.0);
    vec3 color = (t < 0.0) ? mix(cold, mid, t + 1.0) : mix(mid, warm, t);
    f_color = vec4(color, u_opacity);
}
"""


def _quad_verts(x0, y0, x1, y1, with_uv=False):
    if with_uv:
        return [x0, y0, 0.0, 0.0,
                x1, y0, 1.0, 0.0,
                x0, y1, 0.0, 1.0,
                x1, y0, 1.0, 0.0,
                x1, y1, 1.0, 1.0,
                x0, y1, 0.0, 1.0]
    return [x0, y0, x1, y0, x0, y1, x1, y0, x1, y1, x0, y1]


class SnakeRenderer:
    def __init__(self, H: int, W: int, resolution: int = 800, mode: str = "offscreen",
                 head_color=(0.0, 1.0, 0.8), tail_color=(0.0, 0.2, 0.4),
                 food_color=(1.0, 0.42, 0.21), heatmap_opacity: float = 0.4):
        self.H = H
        self.W = W
        self.res = resolution
        self.mode = mode
        self.head_color = np.array(head_color, dtype=np.float32)
        self.tail_color = np.array(tail_color, dtype=np.float32)
        self.food_color = np.array(food_color, dtype=np.float32)
        self.heatmap_opacity = heatmap_opacity
        # Visible wall framing the play field. The framebuffer clears to the
        # wall colour; the darker play field is drawn inset by wall_ndc, so the
        # border the snake dies against is clearly visible.
        self.wall_ndc = 0.05
        self.wall_color = np.array([0.34, 0.30, 0.46], dtype=np.float32)
        self.field_color = np.array([0.05, 0.05, 0.08], dtype=np.float32)
        self._quit = False
        self._pygame_keys = []

        if mode == "offscreen":
            self.ctx = moderngl.create_standalone_context()
            self.fbo = self.ctx.simple_framebuffer((resolution, resolution))
        else:
            import pygame
            pygame.init()
            # Request OpenGL 3.3 Core Profile — required on macOS for moderngl
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,
                                            pygame.GL_CONTEXT_PROFILE_CORE)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)
            pygame.display.set_caption("Snake PPO")
            self._pygame = pygame
            self._screen = pygame.display.set_mode(
                (resolution, resolution), pygame.OPENGL | pygame.DOUBLEBUF
            )
            self.ctx = moderngl.create_context()
            self.fbo = self.ctx.screen

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self._body_prog = self.ctx.program(vertex_shader=_BODY_VERT, fragment_shader=_BODY_FRAG)
        self._food_prog = self.ctx.program(vertex_shader=_FOOD_VERT, fragment_shader=_FOOD_FRAG)
        self._heat_prog = self.ctx.program(vertex_shader=_HEAT_VERT, fragment_shader=_HEAT_FRAG)

        # Pre-allocate VBOs (will orphan/resize as needed)
        max_body = H * W
        self._body_vbo = self.ctx.buffer(reserve=max_body * 6 * 5 * 4)
        self._food_vbo = self.ctx.buffer(reserve=6 * 4 * 4)
        self._heat_vbo = self.ctx.buffer(reserve=H * W * 6 * 3 * 4)

        self._body_vao = self.ctx.vertex_array(
            self._body_prog, [(self._body_vbo, "2f 3f", "in_pos", "in_color")])
        self._food_vao = self.ctx.vertex_array(
            self._food_prog, [(self._food_vbo, "2f 2f", "in_pos", "in_uv")])
        self._heat_vao = self.ctx.vertex_array(
            self._heat_prog, [(self._heat_vbo, "2f 1f", "in_pos", "in_value")])

    # ------------------------------------------------------------------
    # Cell geometry helpers
    # ------------------------------------------------------------------

    def _cell_bounds(self, row, col, gap: float = 0.08):
        # Cells map into the inset play field [-A, A], leaving a wall border.
        A = 1.0 - self.wall_ndc
        x0f = -A + 2.0 * A * col / self.W
        x1f = -A + 2.0 * A * (col + 1) / self.W
        y1f = A - 2.0 * A * row / self.H
        y0f = A - 2.0 * A * (row + 1) / self.H
        gx = (x1f - x0f) * gap / 2
        gy = (y1f - y0f) * gap / 2
        return x0f + gx, y0f + gy, x1f - gx, y1f - gy

    def _draw_solid_quad(self, x0, y0, x1, y1, color):
        c = color
        verts = []
        for px, py in ((x0,y0),(x1,y0),(x0,y1),(x1,y0),(x1,y1),(x0,y1)):
            verts += [px, py, c[0], c[1], c[2]]
        arr = np.array(verts, dtype=np.float32)
        self._body_vbo.orphan(arr.nbytes)
        self._body_vbo.write(arr)
        self._body_vao.render(moderngl.TRIANGLES, vertices=6)

    def _make_body_verts(self, body, dead: bool = False):
        n = len(body)
        head_c = np.array([1.0, 0.15, 0.1], dtype=np.float32) if dead else self.head_color
        tail_c = np.array([0.4, 0.0, 0.0], dtype=np.float32) if dead else self.tail_color
        verts = []
        for i, (row, col) in enumerate(body):
            t = i / max(n - 1, 1)
            c = (1 - t) * head_c + t * tail_c
            x0, y0, x1, y1 = self._cell_bounds(row, col, gap=0.1)
            for px, py in ((x0,y0),(x1,y0),(x0,y1),(x1,y0),(x1,y1),(x0,y1)):
                verts += [px, py, c[0], c[1], c[2]]
        return np.array(verts, dtype=np.float32)

    def _make_food_verts(self, food):
        row, col = food
        x0, y0, x1, y1 = self._cell_bounds(row, col, gap=0.04)
        return np.array(_quad_verts(x0, y0, x1, y1, with_uv=True), dtype=np.float32)

    def _make_heat_verts(self, value_grid):
        vmin, vmax = value_grid.min(), value_grid.max()
        mid = (vmin + vmax) / 2.0
        scale = max(vmax - vmin, 1e-6)
        verts = []
        for row in range(self.H):
            for col in range(self.W):
                v = float((value_grid[row, col] - mid) / scale * 2.0)
                x0, y0, x1, y1 = self._cell_bounds(row, col, gap=0.0)
                for px, py in ((x0,y0),(x1,y0),(x0,y1),(x1,y0),(x1,y1),(x0,y1)):
                    verts += [px, py, v]
        return np.array(verts, dtype=np.float32)

    # ------------------------------------------------------------------
    # Core draw
    # ------------------------------------------------------------------

    def _draw(self, state: dict, time: float = 0.0, value_grid=None, dead: bool = False):
        body = state["body"]
        food = state["food"]

        # Play field inset inside the wall border (border = clear/wall colour).
        A = 1.0 - self.wall_ndc
        field = np.array([0.20, 0.03, 0.03], dtype=np.float32) if dead else self.field_color
        self._draw_solid_quad(-A, -A, A, A, field)

        if body:
            bv = self._make_body_verts(body, dead=dead)
            self._body_vbo.orphan(bv.nbytes)
            self._body_vbo.write(bv)
            self._body_vao.render(moderngl.TRIANGLES, vertices=len(body) * 6)

        fv = self._make_food_verts(food)
        self._food_vbo.orphan(fv.nbytes)
        self._food_vbo.write(fv)
        self._food_prog["u_color"].value = tuple(self.food_color)
        self._food_prog["u_time"].value = float(time)
        self._food_vao.render(moderngl.TRIANGLES, vertices=6)

        if value_grid is not None:
            hv = self._make_heat_verts(value_grid)
            self._heat_vbo.orphan(hv.nbytes)
            self._heat_vbo.write(hv)
            self._heat_prog["u_opacity"].value = self.heatmap_opacity
            self._heat_vao.render(moderngl.TRIANGLES, vertices=self.H * self.W * 6)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_frame(self, state: dict, time: float = 0.0, value_grid=None,
                     dead: bool = False) -> np.ndarray:
        """Offscreen mode — returns RGB uint8 (H_px, W_px, 3)."""
        self.fbo.use()
        self.ctx.clear(*self.wall_color)
        self._draw(state, time, value_grid, dead=dead)
        pixels = self.fbo.read(components=3)
        frame = np.frombuffer(pixels, dtype=np.uint8).reshape(self.res, self.res, 3)
        return np.flipud(frame)

    def show(self, state: dict, time: float = 0.0, value_grid=None, dead: bool = False):
        """Windowed mode — renders to window and swaps buffers."""
        pygame = self._pygame
        self._pygame_keys = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self._quit = True
                self._pygame_keys.append(event.key)

        self.ctx.screen.use()
        self.ctx.clear(*self.wall_color)
        self._draw(state, time, value_grid, dead=dead)
        pygame.display.flip()

    def should_quit(self) -> bool:
        return self._quit

    def last_keys(self) -> list:
        return self._pygame_keys

    def close(self):
        self.ctx.release()
        if self.mode == "window":
            self._pygame.quit()
