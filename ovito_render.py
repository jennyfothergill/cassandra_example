
def ovito_render():
    "Render images using OVITO"
    import ovito
    from ovito.io import import_file
    from ovito.vis import Viewport, RenderSettings, TachyonRenderer
    from ovito.modifiers import ColorCodingModifier
    from ovito.modifiers import PolyhedralTemplateMatchingModifier
    
    pipeline = import_file("gcmc.out.xyz")
    #ptm_modifier = PolyhedralTemplateMatchingModifier(
    #    color_by_type=True,
    #    rmsd_cutoff=0.3,
    #)
    #for structure_type in ['BCC', 'CUBIC_DIAMOND', 'FCC', 'GRAPHENE',
    #                       'HCP', 'HEX_DIAMOND', 'ICO', 'OTHER', 'SC']:
    #    structure_type = getattr(PolyhedralTemplateMatchingModifier.Type, structure_type)
    #    ptm_modifier.structures[structure_type].enabled = True
    #pipeline.modifiers.append(ptm_modifier)
    data_collection = pipeline.compute()
    pipeline.add_to_scene()
    renderer = TachyonRenderer(antialiasing_samples=32,)
    vp = Viewport(type=Viewport.Type.Perspective, camera_dir=(2, 1, -1))
    vp.zoom_all()
    vp.camera_pos = np.asarray(vp.camera_pos) * np.array([0.8, 0.8, 0.75])
    vp.render_image(
        filename='sample.png',
        size=(1000, 750),
        frame=pipeline.source.num_frames-1,
        alpha=True,
        renderer=renderer,
    )
    every_nth = pipeline.source.num_frames // 20
    video = vp.render_anim(
        filename='sample.gif',
        size=(480, 320),
        fps=5,
        renderer=renderer,
        every_nth=every_nth,
    )
    image = vp.render_image(size=(800, 600), filename='img.png', frame=pipeline.source.num_frames)
    pipeline.remove_from_scene()
    

if __name__ == "__main__":
    ovito_render()
