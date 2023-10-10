// https://observablehq.com/@stardisblue/citations@1064
import define1 from "./08873bb244693974@516.js";
import define2 from "./41faeb3ac342a72d@287.js";
import define3 from "./3fb273d7cee36220@544.js";

function _1(md){return(
md`# Citations

\`\`\`observablehq
import { bib } from "@stardisblue/citations"
\`\`\`
`
)}

function _2(cite,md){return(
md`
It will probably extend to something named *Observable Writing Set*, but let's not get ahead of ourselves.

Inspired of _@somethingelseentirely/papers_${cite("somethingelseentirely2021")}, but i was not very satisfied of how things were handled so I made another one 🤓.

Now that presentations are done, let's get into it :).

First we need to create our bibliography. *Note the usage of viewof*`
)}

function _bibliography(bib){return(
bib({
  Tegally2021: `Tegally, H., Wilkinson, E., Giovanetti, M. *et al*. Detection of a SARS-CoV-2 variant of concern in South Africa. *Nature* **592**, 438–443 (2021). https://doi.org/10.1038/s41586-021-03402-9`,
  Chen2020: `F. Chen, L. Piccinini, P. Poncelet, and A. Sallaberry. Node Overlap Removal Algorithms: An Extended Comparative Study. *In Journal of Graph Algorithms and Applications*, Vol. 24, no. 4, pp. 683-706, 2020. Regular paper. http://doi.org/10.7155/jgaa.00532`,
  somethingelseentirely2021: `somethingelseentirely. Papers. *Observable*. Website. https://observablehq.com/@somethingelseentirely/papers`
})
)}

function _4(md){return(
md`To make our life easier, we will store the \`viewof\` as another variable (\`cite\`)`
)}

function _cite($0){return(
$0
)}

function _6(tex,cite,$0,md){return(
md`You can now use \`cite\` to mark the references (as in ${tex`\LaTeX`}): Here is an example \`cite("Tegally2021")\`${cite("Tegally2021")}. You can use it multiple times.
I use this${$0("Tegally2021")} paper because I was inspired to do a citation system thanks to it.

**Update**
You can now cite multiple papers: \`cite("Tegally2021", "somethingelseentirely2021")\`${cite("Tegally2021", "somethingelseentirely2021")}

All is left is to display the bibliography, to do so simply do :`
)}

function _7(bibliography){return(
bibliography
)}

function _8(md){return(
md`As you can see, \`chen2020\` *shameless ad* is not displayed. Because it is not used anywhere... *not so shameless afterall*.

It will also update on the fly thanks to the viewof value combination.`
)}

function _9(md){return(
md`## Feedback

All feedback is welcomed (critisism and propositions 🎉).`
)}

function _10(md){return(
md`## Documentation`
)}

function _bib(frepr,defaultCitation,defaultEntry,AdvancedSetter,html,bibdoc){return(
frepr(function (entries, options = {}) {
  const {
    numbered = true,
    citation = defaultCitation,
    entry = defaultEntry
  } = options;

  return new AdvancedSetter(new Map(), {
    set(state, ...keys) {
      return citation(
        keys.map((key) => {
          const ref = entries[key];
          if (ref === undefined)
            throw new Error(`${key} was not found in given bibliography`);

          // const [id, render] = state.get(key); return citation(id, render);
          if (state.has(key)) return [...state.get(key)];

          const id = numbered ? state.size : key,
            pair = [id, entry(ref, id)];
          state.set(key, pair);

          return pair;
        })
      );
    },
    get(state) {
      return html`${Array.from(state.values(), ([, rendered]) => rendered)}`;
    }
  });
}, bibdoc)
)}

function _bibdoc(md){return(
{
  name: `bib`,
  description: md`Creates a bibliography.
- \`entries\`: Object, containing the \`key: value\` bibliography **required**
- \`options\`: Object, allows for bibliography customization (see below).

Options:
- \`numbered\`: boolean, shows numbers instead keys. default \`true\`.
- \`citation\`: function, controls how a citation is displayed. Recieves _id_ the key (or number) to display and _to_ the entry \`HTMLElement\` (useful for linking). returns the rendered object.

  default \`defaultCitation\`

- \`entry\`: function, controls how the item is displayed in the bibliography. Recieves _value_, entries' value and _id_ the key (or number if it is numbered). returns the rendered object.

  default \`defaultEntry\`

  
`
}
)}

function _13(md){return(
md`---`
)}

function _defaultCitation(html,addCommas,link){return(
(citations) =>
  html`[${addCommas(citations.map(([id, to]) => link(id, to)))}]`
)}

function _defaultEntry(html,md){return(
(value, id) => html`<div id="ref-${id}">${md`[${id}]  ${value}`}`
)}

function _addCommas(){return(
(list, separator = ", ") => {
  const newList = [];
  list.forEach((v, i) => {
    if (i !== 0) newList.push(separator);
    newList.push(v);
  });
  return newList;
}
)}

function _17(md){return(
md`## Imports`
)}

function _19(md){return(
md`As always thanks [@mootari](https://observablehq.com/@mootari) for [@mootari/signature](https://observablehq.com/@mootari/signature)`
)}

export default function define(runtime, observer) {
  const main = runtime.module();
  main.variable(observer()).define(["md"], _1);
  main.variable(observer()).define(["cite","md"], _2);
  main.variable(observer("viewof bibliography")).define("viewof bibliography", ["bib"], _bibliography);
  main.variable(observer("bibliography")).define("bibliography", ["Generators", "viewof bibliography"], (G, _) => G.input(_));
  main.variable(observer()).define(["md"], _4);
  main.variable(observer("cite")).define("cite", ["viewof bibliography"], _cite);
  main.variable(observer()).define(["tex","cite","viewof bibliography","md"], _6);
  main.variable(observer()).define(["bibliography"], _7);
  main.variable(observer()).define(["md"], _8);
  main.variable(observer()).define(["md"], _9);
  main.variable(observer()).define(["md"], _10);
  main.variable(observer("viewof bib")).define("viewof bib", ["frepr","defaultCitation","defaultEntry","AdvancedSetter","html","bibdoc"], _bib);
  main.variable(observer("bib")).define("bib", ["Generators", "viewof bib"], (G, _) => G.input(_));
  main.variable(observer("bibdoc")).define("bibdoc", ["md"], _bibdoc);
  main.variable(observer()).define(["md"], _13);
  main.variable(observer("defaultCitation")).define("defaultCitation", ["html","addCommas","link"], _defaultCitation);
  main.variable(observer("defaultEntry")).define("defaultEntry", ["html","md"], _defaultEntry);
  main.variable(observer("addCommas")).define("addCommas", _addCommas);
  main.variable(observer()).define(["md"], _17);
  const child1 = runtime.module(define1);
  main.import("AdvancedSetter", child1);
  main.variable(observer()).define(["md"], _19);
  const child2 = runtime.module(define2);
  main.import("frepr", child2);
  const child3 = runtime.module(define3);
  main.import("link", child3);
  return main;
}
