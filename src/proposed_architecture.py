proposed_architecture

train:
    word_to_index, embeddings = WordEmbeddingLoader.load(params)
    BOW = BagOfWords(embeddings, word_to_index)

    model = Feedforward(params, BOW)

    model = FNN2(parametes, BOW)
    pytorch.save(model, 'some_path')

test:
    model = pytorch.load('some_path')
    embedder = model.embedder

    sentence = embedder.to_vector('some sentence')
    model(sentece)
