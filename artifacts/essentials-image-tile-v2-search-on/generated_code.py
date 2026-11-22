"""V3 node definitions for the draft_grid_tiler pack (exactly two nodes)."""

from __future__ import annotations

from comfy_api.latest import io

from .grid_core import (
    BLEND_MODES, DEFAULT_BLEND, MAX_GRID, MAX_OVERLAP, tile_image, untile_image,
)

TILE_NODE_ID = "DraftGridTiler_ImageGridTile"
UNTILE_NODE_ID = "DraftGridTiler_ImageGridUntile"


def _grid_widgets():
    # Stable widget order shared by both nodes: rows, columns, overlap_x, overlap_y.
    return [
        io.Int.Input("rows", default=2, min=1, max=MAX_GRID, step=1,
                     tooltip="Tile rows (height axis)."),
        io.Int.Input("columns", default=2, min=1, max=MAX_GRID, step=1,
                     tooltip="Tile columns (width axis)."),
        io.Int.Input("overlap_x", default=0, min=0, max=MAX_OVERLAP, step=1,
                     tooltip="Horizontal pixel overlap between neighbouring columns; < tile width."),
        io.Int.Input("overlap_y", default=0, min=0, max=MAX_OVERLAP, step=1,
                     tooltip="Vertical pixel overlap between neighbouring rows; < tile height."),
    ]


class DraftGridTilerImageGridTile(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=TILE_NODE_ID,
            display_name="Image Grid Tile (draft)",
            category="image/draft_grid_tiler",
            description=(
                "Split each image of a BHWC batch into a rows x columns grid of equal tiles. "
                "Per axis: extent*count - overlap*(count-1) == dimension (overlap_x for width, "
                "overlap_y for height). Order: batch-major, row-major."
            ),
            inputs=[io.Image.Input("image")] + _grid_widgets(),
            outputs=[
                io.Image.Output(display_name="tiles"),
                io.Int.Output(display_name="original_width"),
                io.Int.Output(display_name="original_height"),
                io.Int.Output(display_name="source_batch_size"),
            ],
        )

    @classmethod
    def execute(cls, image, rows, columns, overlap_x, overlap_y) -> io.NodeOutput:
        tiles, width, height, batch = tile_image(image, rows, columns, overlap_x, overlap_y)
        return io.NodeOutput(tiles, width, height, batch)


class DraftGridTilerImageGridUntile(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=UNTILE_NODE_ID,
            display_name="Image Grid Untile (draft)",
            category="image/draft_grid_tiler",
            description=(
                "Rebuild the source batch from Image Grid Tile output. Validates exact tile count and "
                "size; overlap bands are blended with linear or cosine edge ramps normalised by the "
                "accumulated weight; overlap_x = overlap_y = 0 is byte-exact."
            ),
            inputs=[io.Image.Input("tiles")] + _grid_widgets() + [
                io.Combo.Input("blend", options=list(BLEND_MODES), default=DEFAULT_BLEND,
                               tooltip="Edge ramp used across overlap bands."),
                io.Int.Input("original_width", default=64, min=1, max=16384, force_input=True),
                io.Int.Input("original_height", default=64, min=1, max=16384, force_input=True),
                io.Int.Input("source_batch_size", default=1, min=1, max=4096, force_input=True),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, tiles, rows, columns, overlap_x, overlap_y, blend,
                original_width, original_height, source_batch_size) -> io.NodeOutput:
        image = untile_image(tiles, rows, columns, overlap_x, overlap_y,
                             original_width, original_height, source_batch_size, blend)
        return io.NodeOutput(image)


NODE_CLASSES = [DraftGridTilerImageGridTile, DraftGridTilerImageGridUntile]
