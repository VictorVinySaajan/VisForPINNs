// https://observablehq.com/@stardisblue/repr@287
import define1 from "./3d9d1394d858ca97@553.js";

function _1(md){return(
md`# \`repr\`

A little helper for informative representations`
)}

function _repr(representation){return(
representation(
  `~~~observable
import { repr } from "@stardisblue/repr"
~~~`,
  representation
)
)}

function _3(md){return(
md`**Example**`
)}

function _norm(repr,tex){return(
repr(
  tex`\text{norm}(a, b) = ||c_a - c_b||`,
  ([ax, ay], [bx, by]) => Math.hypot(ax - bx, ay - by)
)
)}

function _5(norm){return(
norm([0, 0], [10, 10])
)}

function _6(md){return(
md`**Documentation**`
)}

function _7(frepr,repr){return(
frepr(repr, {
  name: "repr",
  description: `
- \`repr <string | any>\`: is rendered using \`md\` if \`string\`, passed as is otherwise.
- \`value <any>\`: &mdash; assigned as value. `
})
)}

function _representation(md){return(
(repr, value) =>
  Object.assign(typeof repr === "string" ? md`${repr}` : repr, {
    value
  })
)}

function _9(md){return(
md`---`
)}

function _10(md){return(
md` ## \`frepr\``
)}

function _11(md){return(
md`A little bonus, using [@mootari](https://observablehq.com/@mootari)'s \`signature\` function`
)}

function _frepr(repr,functionRepresentation){return(
repr(
  `~~~observable
import { frepr } from "@stardisblue/repr"
~~~`,
  functionRepresentation
)
)}

function _13(md){return(
md`**Example**`
)}

function _sum(frepr){return(
frepr(
  function sum(a, b) {
    return a + b;
  },
  { description: "**returns** the sum of two numbers" }
)
)}

function _15(sum){return(
sum(12, 3)
)}

function _16(md){return(
md`**Documentation**`
)}

function _17(signature,frepr){return(
signature(frepr, {
  name: "frepr",
  description: `
Represent the signature and holds the function as a value

See [@mootari/signature](https://observablehq.com/@mootari/signature#signature) for more information about parameters`
})
)}

function _functionRepresentation(signature){return(
(fn, options) => {
  const copy = { ...options };
  if (copy.tests)
    copy.tests = Object.fromEntries(
      Object.entries(copy.tests).map(([key, value]) => [
        key,
        (assert) => value(assert, fn)
      ])
    );

  return Object.assign(signature(fn, copy), { value: fn });
}
)}

export default function define(runtime, observer) {
  const main = runtime.module();
  main.variable(observer()).define(["md"], _1);
  main.variable(observer("viewof repr")).define("viewof repr", ["representation"], _repr);
  main.variable(observer("repr")).define("repr", ["Generators", "viewof repr"], (G, _) => G.input(_));
  main.variable(observer()).define(["md"], _3);
  main.variable(observer("viewof norm")).define("viewof norm", ["repr","tex"], _norm);
  main.variable(observer("norm")).define("norm", ["Generators", "viewof norm"], (G, _) => G.input(_));
  main.variable(observer()).define(["norm"], _5);
  main.variable(observer()).define(["md"], _6);
  main.variable(observer()).define(["frepr","repr"], _7);
  main.variable(observer("representation")).define("representation", ["md"], _representation);
  main.variable(observer()).define(["md"], _9);
  main.variable(observer()).define(["md"], _10);
  main.variable(observer()).define(["md"], _11);
  main.variable(observer("viewof frepr")).define("viewof frepr", ["repr","functionRepresentation"], _frepr);
  main.variable(observer("frepr")).define("frepr", ["Generators", "viewof frepr"], (G, _) => G.input(_));
  main.variable(observer()).define(["md"], _13);
  main.variable(observer("viewof sum")).define("viewof sum", ["frepr"], _sum);
  main.variable(observer("sum")).define("sum", ["Generators", "viewof sum"], (G, _) => G.input(_));
  main.variable(observer()).define(["sum"], _15);
  main.variable(observer()).define(["md"], _16);
  main.variable(observer()).define(["signature","frepr"], _17);
  main.variable(observer("functionRepresentation")).define("functionRepresentation", ["signature"], _functionRepresentation);
  const child1 = runtime.module(define1);
  main.import("signature", child1);
  return main;
}
