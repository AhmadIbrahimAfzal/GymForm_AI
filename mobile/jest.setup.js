// Fix Jest VM realm TypedArray instanceof checks for native C++ addons like onnxruntime-node
// Native C++ addons return TypedArrays from the outer V8 realm, which fail Jest VM's `instanceof` check.
// Patching Symbol.hasInstance on TypedArrays makes cross-realm instanceof checks work seamlessly.

const typedArrayClasses = [
  Float32Array,
  Float64Array,
  Int8Array,
  Int16Array,
  Int32Array,
  Uint8Array,
  Uint8ClampedArray,
  Uint16Array,
  Uint32Array,
];

for (const cls of typedArrayClasses) {
  Object.defineProperty(cls, Symbol.hasInstance, {
    value: function (instance) {
      return (
        instance !== null &&
        typeof instance === 'object' &&
        (instance.constructor.name === cls.name ||
          Object.prototype.toString.call(instance) === `[object ${cls.name}]`)
      );
    },
    writable: true,
    configurable: true,
  });
}
