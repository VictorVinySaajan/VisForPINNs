import define1 from "./115f2c4ec1a42d7e@373.js";

function _1(md){return(
md`# NumPy library for Observable

The NumPy Python package for Javascript and Observable via [Pyodide](https://observablehq.com/@gnestor/pyodide).

## Usage

~~~js
import { numpy as np } from '@gnestor/numpy'
~~~

~~~js
np.dot(np.array([2, 1, 5, 4]), np.array([3, 4, 7, 8]))
~~~

See the [Numpy Demo](https://observablehq.com/@gnestor/numpy-demo) for usage examples and [NumPy docs](https://numpy.org/doc/stable/reference/index.html) for API reference `
)}

async function _numpy(pyodide,py)
{
  await pyodide.loadPackage("numpy");
  return py`import numpy
numpy`;
}


export default function define(runtime, observer) {
  const main = runtime.module();
  main.variable(observer()).define(["md"], _1);
  main.variable(observer("numpy")).define("numpy", ["pyodide","py"], _numpy);
  const child1 = runtime.module(define1);
  main.import("py", child1);
  main.import("pyodide", child1);
  return main;
}
