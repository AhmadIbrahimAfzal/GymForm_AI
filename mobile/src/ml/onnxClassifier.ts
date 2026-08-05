interface OrtTensor {
  data: any;
}

interface OrtSession {
  run(feeds: Record<string, unknown>): Promise<Record<string, OrtTensor>>;
}

export interface OrtLike {
  InferenceSession: {
    create(pathOrBuffer: any): Promise<OrtSession>;
  };
  Tensor: new (type: 'float32', data: Float32Array, dims: number[]) => unknown;
}

export interface OnnxClassifier {
  run(input: Float32Array): Promise<Float32Array>;
}

export async function createOnnxClassifier(
  modelPathOrBuffer: any,
  ort: OrtLike,
): Promise<OnnxClassifier> {
  const session = await ort.InferenceSession.create(modelPathOrBuffer);
  return {
    async run(input: Float32Array): Promise<Float32Array> {
      const feeds = { angles: new ort.Tensor('float32', input, [1, 8]) };
      const out = await session.run(feeds);
      return Float32Array.from(out['logits'].data as Iterable<number>);
    },
  };
}
