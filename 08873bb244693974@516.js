// https://observablehq.com/@stardisblue/setter@516
import define1 from "./41faeb3ac342a72d@287.js";
import define2 from "./3d9d1394d858ca97@553.js";
import define3 from "./a2e58f97fd5e8d7c@754.js";

function _1(md){return(
md`# My Take On Exercice Notebooks`
)}

function _2(md){return(
md`~~~observable
import {Setter} from "@stardisblue/setter"
~~~`
)}

function _3(md){return(
md`**tl;dr**:`
)}

function _4(md,answer){return(
md`

> **Exercice 1**
>
> Set x such as \`x - 10 = 100\`
>
>  *Your Answer*:
>
> <span style="color: ${answer === 90 ? "" : "red"}">\`${
  answer ?? "x"
} - 10 = 100\` ${answer === 90 ? "🎉" : "😟"}
`
)}

function _6(md){return(
md`## Wait, how does it work ?? `
)}

function _answer(Setter){return(
Setter(null)
)}

function _8($0,x){return(
$0(x)
)}

function _9(md){return(
md`Looks weird but simple no ?`
)}

function _10(md){return(
md`## Can we do more ?`
)}

function _11(md){return(
md`I actually implemented setter to be able to be any structure you want :)`
)}

function _htmlExample(Setter,md){return(
Setter("Stranger", {
  get: (value) => md`Hello ${value}`
})
)}

function _13(htmlExample){return(
htmlExample
)}

function _15($0,yourname){return(
$0(yourname)
)}

function _16(md){return(
md`---`
)}

function _17(md){return(
md`Instead of one value per Setter you can change it to be a list`
)}

function _addValue(Button){return(
Button("Add value")
)}

function _19(map){return(
map
)}

function _20(addValue,$0)
{
  addValue,
    $0(
      "item" + ((Math.random() * 100) | 0),
      "value" + ((Math.random() * 100) | 0)
    );
}


function _map(Setter,md){return(
Setter(new Map(), {
  set(map, key, value) {
    map.set(key, value);
    return map;
  },
  get(map) {
    return md`${Array.from(map, ([k, v]) => `- **${k}**: ${v}\n`)}`;
  }
})
)}

function _22(md){return(
md`thank you for reading, I'm not very good at writing stuff, feedback is very welcome :)`
)}

function _23(signature,Setter,md){return(
signature(Setter, {
  description: md`Struggling to find something to put here...

~~~observablehq
import {Setter} from "@stardisblue/setter"
~~~`
})
)}

function _Setter(EventTarget){return(
function (
  value,
  {
    set = (state, ...[value]) => value /* used to set the value */,
    get = (value) => value /* used to retrieve the value */
  } = {}
) {
  const input = new EventTarget();
  const ref = { value };

  const ƒ = function (...value) {
    ref.value = set(ref.value, ...value);
    ƒ.value = get(ref.value);
    input.dispatchEvent(new CustomEvent("input"));
  };

  // raw way to mascarade ƒ as an eventtarget
  ƒ.addEventListener = (...args) => input.addEventListener(...args);
  ƒ.dispatchEvent = (...args) => input.dispatchEvent(...args);
  ƒ.removeEventListener = (...args) => input.removeEventListener(...args);
  ƒ.value = get(ref.value);

  return ƒ;
}
)}

function _25(md){return(
md`APPENDIX

You can also create this behaviour using an [observer](https://github.com/observablehq/stdlib/blob/master/README.md#Generators_observe) but at the cost of a weird inversion of control and a single use value.`
)}

function _test(ObserverSetter){return(
ObserverSetter(null)
)}

function _27(test){return(
test.value
)}

function _29(test,a){return(
test(a)
)}

function _ObserverSetter(Generators){return(
function (
  value,
  {
    set = (state, value) => value /* used to set the value */,
    get = (value) => value /* used to retrieve the value */
  } = {}
) {
  const ref = { value };
  let callback;

  const ƒ = function (value) {
    ref.value = set(ref.value, value);
    return callback(get(ref.value));
  };

  ƒ.value = Generators.observe((change) => {
    callback = (v) => change(v);
    change(get(ref.value));
  });

  return ƒ;
}
)}

function _31(md){return(
md`**WIP**`
)}

function _AdvancedSetter(frepr,EventTarget){return(
frepr(function AdvancedSetter(
  value,
  {
    set = function (state, value) {
      this.value = value;
    } /* used to set the value */,
    get = (value) => value /* used to retrieve the value */
  } = {}
) {
  const input = new EventTarget();
  const ref = { value };

  const ƒ = function (...value) {
    const result = set.bind(ref)(ref.value, ...value);
    ƒ.value = get(ref.value);
    input.dispatchEvent(new CustomEvent("input"));
    return result;
  };

  // raw way to mascarade ƒ as an eventtarget
  ƒ.addEventListener = (...args) => input.addEventListener(...args);
  ƒ.dispatchEvent = (...args) => input.dispatchEvent(...args);
  ƒ.removeEventListener = (...args) => input.removeEventListener(...args);
  ƒ.value = get(ref.value);

  return ƒ;
})
)}

function _33(md){return(
md`## Imports`
)}

export default function define(runtime, observer) {
  const main = runtime.module();
  main.variable(observer()).define(["md"], _1);
  main.variable(observer()).define(["md"], _2);
  main.variable(observer()).define(["md"], _3);
  main.variable(observer()).define(["md","answer"], _4);
  main.variable(observer()).define(["md"], _6);
  main.variable(observer("viewof answer")).define("viewof answer", ["Setter"], _answer);
  main.variable(observer("answer")).define("answer", ["Generators", "viewof answer"], (G, _) => G.input(_));
  main.variable(observer()).define(["viewof answer","x"], _8);
  main.variable(observer()).define(["md"], _9);
  main.variable(observer()).define(["md"], _10);
  main.variable(observer()).define(["md"], _11);
  main.variable(observer("viewof htmlExample")).define("viewof htmlExample", ["Setter","md"], _htmlExample);
  main.variable(observer("htmlExample")).define("htmlExample", ["Generators", "viewof htmlExample"], (G, _) => G.input(_));
  main.variable(observer()).define(["htmlExample"], _13);
  main.variable(observer()).define(["viewof htmlExample","yourname"], _15);
  main.variable(observer()).define(["md"], _16);
  main.variable(observer()).define(["md"], _17);
  main.variable(observer("viewof addValue")).define("viewof addValue", ["Button"], _addValue);
  main.variable(observer("addValue")).define("addValue", ["Generators", "viewof addValue"], (G, _) => G.input(_));
  main.variable(observer()).define(["map"], _19);
  main.variable(observer()).define(["addValue","viewof map"], _20);
  main.variable(observer("viewof map")).define("viewof map", ["Setter","md"], _map);
  main.variable(observer("map")).define("map", ["Generators", "viewof map"], (G, _) => G.input(_));
  main.variable(observer()).define(["md"], _22);
  main.variable(observer()).define(["signature","Setter","md"], _23);
  main.variable(observer("Setter")).define("Setter", ["EventTarget"], _Setter);
  main.variable(observer()).define(["md"], _25);
  main.variable(observer("test")).define("test", ["ObserverSetter"], _test);
  main.variable(observer()).define(["test"], _27);
  main.variable(observer()).define(["test","a"], _29);
  main.variable(observer("ObserverSetter")).define("ObserverSetter", ["Generators"], _ObserverSetter);
  main.variable(observer()).define(["md"], _31);
  main.variable(observer("viewof AdvancedSetter")).define("viewof AdvancedSetter", ["frepr","EventTarget"], _AdvancedSetter);
  main.variable(observer("AdvancedSetter")).define("AdvancedSetter", ["Generators", "viewof AdvancedSetter"], (G, _) => G.input(_));
  main.variable(observer()).define(["md"], _33);
  const child1 = runtime.module(define1);
  main.import("frepr", child1);
  const child2 = runtime.module(define2);
  main.import("signature", child2);
  const child3 = runtime.module(define3);
  main.import("Button", child3);
  return main;
}
