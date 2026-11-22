"""V3 node definitions for the TileGrid Kit draft pack (unpublished)."""

from __future__ import annotations

from comfy_api.latest import ComfyExtension, io

from .tilegrid_core import (
    BLEND_MODES,
    DEFAULT_BLEND,
    MAX_GRID,
    MAX_OVERLAP,
    tile_image,
    untile_image,
)

CATEGORY = "image/tilegrid_kit"


class TileGridKitImageGridTile(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TileGridKit_ImageGridTile",
            display_name="Image Grid Tile (TileGrid Kit)",
            category=CATEGORY,
            description=(
                "Splits every image of a BHWC batch into rows x columns equal tiles. Neighbouring "
                "tiles share `overlap_x` pixels horizontally and `overlap_y` pixels vertically. "
                "Output order: source batch, then row-major. Requires "
                "extent*count - overlap*(count-1) == dimension on each axis."
            ),
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("rows", default=2, min=1, max=MAX_GRID, step=1),
                io.Int.Input("columns", default=2, min=1, max=MAX_GRID, step=1),
                io.Int.Input("overlap_x", default=0, min=0, max=MAX_OVERLAP, step=1,
                             tooltip="Pixels shared by horizontally neighbouring tiles."),
                io.Int.Input("overlap_y", default=0, min=0, max=MAX_OVERLAP, step=1,
                             tooltip="Pixels shared by vertically neighbouring tiles."),
            ],
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


class TileGridKitImageGridUntile(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TileGridKit_ImageGridUntile",
            display_name="Image Grid Untile (TileGrid Kit)",
            category=CATEGORY,
            description=(
                "Rebuilds the original BHWC batch from Image Grid Tile output. Validates the exact "
                "tile count and tile size. Overlap pixels are blended with separable per-axis "
                "edge ramps (linear or half-cosine) normalised by the accumulated weight; with no "
                "overlap the rebuild is an exact copy."
            ),
            inputs=[
                io.Image.Input("tiles"),
                io.Int.Input("original_width", default=64, min=1, max=16384, step=1),
                io.Int.Input("original_height", default=64, min=1, max=16384, step=1),
                io.Int.Input("source_batch_size", default=1, min=1, max=4096, step=1),
                io.Int.Input("rows", default=2, min=1, max=MAX_GRID, step=1),
                io.Int.Input("columns", default=2, min=1, max=MAX_GRID, step=1),
                io.Int.Input("overlap_x", default=0, min=0, max=MAX_OVERLAP, step=1),
                io.Int.Input("overlap_y", default=0, min=0, max=MAX_OVERLAP, step=1),
                io.Combo.Input("blend", options=list(BLEND_MODES), default=DEFAULT_BLEND,
                               tooltip="Edge ramp used inside overlap bands."),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, tiles, original_width, original_height, source_batch_size,
                rows, columns, overlap_x, overlap_y, blend) -> io.NodeOutput:
        image = untile_image(tiles, original_width, original_height, source_batch_size,
                             rows, columns, overlap_x, overlap_y, blend)
        return io.NodeOutput(image)


class TileGridKitExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [TileGridKitImageGridTile, TileGridKitImageGridUntile]


async def comfy_entrypoint() -> TileGridKitExtension:
    return TileGridKitExtension()
