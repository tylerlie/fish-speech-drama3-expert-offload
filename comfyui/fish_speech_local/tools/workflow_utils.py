def to_prompt(graph):
    links={edge[0]:edge for edge in graph['links']};prompt={}
    for node in graph['nodes']:
        inputs={}
        for slot,port in enumerate(node.get('inputs',[])):
            if port.get('link') is not None:
                _,source,source_slot,target,target_slot,kind=links[port['link']]
                assert target==node['id'] and target_slot==slot and kind==port['type']
                inputs[port['name']]=[str(source),source_slot]
        kind,values=node['type'],node.get('widgets_values',[])
        if kind=='DramaReference':inputs.update(audio=values[0],text=values[1])
        elif kind=='PrimitiveStringMultiline':inputs['value']=values[0]
        elif kind in ('DramaSingleVoice','DramaMultiVoice'):
            inputs['seed']=values[0]
            if kind=='DramaSingleVoice' and len(values)>2:inputs['regenerate_rounds']=values[2]
        elif kind=='DramaLoopReference':
            inputs.update(seed=values[0],gap_ms=values[2])
            if len(values)>3:inputs['regenerate_rounds']=values[3]
        elif kind=='DramaAPIParameters':
            # Import by package path when called from the installed tools directory.
            import sys
            from pathlib import Path
            sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
            from fish_speech_local.parameters import PARAMETER_INPUTS
            if len(values)!=len(PARAMETER_INPUTS):raise ValueError('API 参数节点的控件数量不匹配。')
            inputs.update(zip(PARAMETER_INPUTS,values))
        elif kind=='SaveAudioAdvanced':inputs.update(filename_prefix=values[0],format=values[1])
        elif kind not in ('DramaReferences','PreviewAudio'):raise ValueError(kind)
        prompt[str(node['id'])]={'class_type':kind,'inputs':inputs,'_meta':{'title':node.get('title',kind)}}
    return prompt
