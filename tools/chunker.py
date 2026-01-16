def chunk_transcript(segments, chunk_size=60):
    chunks = []
    current_chunk = []
    start_time = segments[0]['start']


    for seg in segments:
        current_chunk.append(seg['text'])
        if seg['end'] - start_time >= chunk_size:
            chunks.append({
            "start": start_time,
            "end": seg['end'],
            "text": " ".join(current_chunk)
            })
            current_chunk = []
            start_time = seg['end']

    if current_chunk:
        chunks.append({
            "start": start_time,
            "end": segments[-1]['end'],
            "text": " ".join(current_chunk)
        })

    return chunks