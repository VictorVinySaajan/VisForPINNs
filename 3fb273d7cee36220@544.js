// https://observablehq.com/@stardisblue/styled-observable@544
import define1 from "./3d9d1394d858ca97@553.js";

function _1(md){return(
md`# Styled Observable`
)}

function _2(md){return(
md`A small list of functions to make my notebook more read friendly`
)}

function _3(md){return(
md`## Definition`
)}

function _4(signature,definition,md,link,_grid){return(
signature(definition, {
  name: "definition",
  description: md`Creates two columns one is treated as code, the other as markdown.

${definition({
  title: "#### Usage",
  code: `definition({
    title:"#### Example", 
    code: "console.log('hello world')",
    lang: "js",
    align: "stretch",
    description: "prints hello world"
})`,
  lang: "js",
  description: md`**Note**: \`code\` and \`description\` must be provided.

Available \`options\`:

- \`title\`: Rendered as markdown, default : \`#### Example\`,
- \`lang\`: Used for code highlight, default : \`json\`,
- \`align\`: string, _see ${link("grid", _grid)}_. default: \`stretch\`,
- \`code\`: Enclosed in markdown highlighted code.,
- \`description\`: Rendered as markdown if string, passed through otherwise`
})}`
})
)}

function _definition(md,html,grid,code){return(
(options = {}) => {
  let { title, lang, code: c, align, description } = {
    title: "#### Example",
    lang: "json",
    align: "stretch",
    ...options
  };

  if (title === undefined) throw new Error('options.title was not defined');
  if (c === undefined) throw new Error('options.code was not defined');
  if (description === undefined)
    throw new Error('options.description was not defined');
  if (typeof description === "string") description = md`${description}`;
  return html`${md`${title}`}${grid([code(c, lang), description], align)}`;
}
)}

function _6(md){return(
md`## Code`
)}

function _7(signature,code,md,definition){return(
signature(code, {
  name: "code",
  description: md`A simple wrapper around the default Markdown codeblock. With added styles

${definition({
  title: "#### Usage",
  code: `code("console.log(hello world)", "js");`,
  lang: "js",
  description: md`Renders the passed \`code\` attributes as a markdown codeblock. \`lang\` is used for highlighting. `
})}`
})
)}

function _code(md,theme){return(
(code, lang = "") => {
  const c = md`~~~${lang}\n${code}\n~~~`;
  c.classList.add(theme);
  return c;
}
)}

function __grid(md){return(
md`## Grid`
)}

function _10(signature,grid,md,definition){return(
signature(grid, {
  name: "grid",
  description: md`Creates a css grid with \`items.length\` columns, automatically wraps when columns' width is below \`16em\`.

${definition({
  title: "#### Usage",
  code: `grid(["column1", "column2"], "stretch");`,
  lang: `js`,
  description: `- \`items\`: array, is passed as-is into observable's \`html\` tag
- \`align\`: is passed into css \`align-items\`, usually \`center\` or \`stretch\`. _see [mdn](https://developer.mozilla.org/en-US/docs/Web/CSS/align-items) for more information_`
})}`
})
)}

function _grid(html){return(
(items, align = "center") =>
  html`<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(16em, 1fr)); align-items: ${align}; column-gap: .5em; ">${items}`
)}

function _12(md){return(
md`## Blockquote`
)}

function _13(signature,blockquote,md){return(
signature(blockquote, {
  name: "blockquote",
  description: md`Rendered using \`md\`. A lazy way to create blockquotes. Created to avoid \`>\` indentation hell, like when you want a codeblock inside. `
})
)}

function _blockquote(md){return(
html => md`> ${html}`
)}

function _15(md){return(
md`## Remark`
)}

function _16(signature,remark,md){return(
signature(remark, {
  name: "remark",
  description: md`When you have something to say that might be important.

${remark(md`Rendered using \`html\`.`)}`
})
)}

function _remark(html){return(
text =>
  html`<div style="padding-left: 20px; padding-right: 20px; margin-left: 20px; margin-right:20px; margin-bottom: 20px; background-color: #f4f4f4;">${text}`
)}

function _18(md){return(
md`## Arrow`
)}

function _19(signature,arrow,md){return(
signature(arrow, {
  name: "arrow",
  description: md`Just a tex arrow: ${arrow()}`
})
)}

function _arrow(tex){return(
() => tex`\rightarrow`
)}

function _21(md){return(
md`## Details`
)}

function _22(signature,details){return(
signature(details, {
  name: "details",
  description: `A lazy way around \`<details>\` to create dropdown content.`
})
)}

function _details(html){return(
(summary, content) =>
  html`<details><summary>${summary}</summary>${content}`
)}

function _24(md){return(
md`## Link`
)}

function _25(signature,link){return(
signature(link, {
  name: "link",
  description: `Sometimes, links inside an notebook does not work. This was created as makeshift solution. It requires \`h\` to be an \`HTMLElement\` with \`id\` attribute.`
})
)}

function _link(md){return(
(title, h) =>
  Object.assign(md`[${title}](#${h.id})`.firstChild, {
    onclick: e => (e.preventDefault(), h.scrollIntoView())
  })
)}

function _27(md){return(
md`## Styles and Imports`
)}

function _theme(DOM){return(
DOM.uid('style-scope').id
)}

function _themeCSS(html,theme){return(
html`<style>
pre.${theme}, .${theme} code {background-color: NavajoWhite; border-radius: 3px;}
pre.${theme} > code, .${theme} pre > code {padding-left: 0; padding-right: 0;}
.${theme} code {padding-left: 0.25em; padding-right: 0.25em;}
pre.${theme}, .${theme} pre {padding-left: 0.5em}`
)}

export default function define(runtime, observer) {
  const main = runtime.module();
  main.variable(observer()).define(["md"], _1);
  main.variable(observer()).define(["md"], _2);
  main.variable(observer()).define(["md"], _3);
  main.variable(observer()).define(["signature","definition","md","link","_grid"], _4);
  main.variable(observer("definition")).define("definition", ["md","html","grid","code"], _definition);
  main.variable(observer()).define(["md"], _6);
  main.variable(observer()).define(["signature","code","md","definition"], _7);
  main.variable(observer("code")).define("code", ["md","theme"], _code);
  main.variable(observer("_grid")).define("_grid", ["md"], __grid);
  main.variable(observer()).define(["signature","grid","md","definition"], _10);
  main.variable(observer("grid")).define("grid", ["html"], _grid);
  main.variable(observer()).define(["md"], _12);
  main.variable(observer()).define(["signature","blockquote","md"], _13);
  main.variable(observer("blockquote")).define("blockquote", ["md"], _blockquote);
  main.variable(observer()).define(["md"], _15);
  main.variable(observer()).define(["signature","remark","md"], _16);
  main.variable(observer("remark")).define("remark", ["html"], _remark);
  main.variable(observer()).define(["md"], _18);
  main.variable(observer()).define(["signature","arrow","md"], _19);
  main.variable(observer("arrow")).define("arrow", ["tex"], _arrow);
  main.variable(observer()).define(["md"], _21);
  main.variable(observer()).define(["signature","details"], _22);
  main.variable(observer("details")).define("details", ["html"], _details);
  main.variable(observer()).define(["md"], _24);
  main.variable(observer()).define(["signature","link"], _25);
  main.variable(observer("link")).define("link", ["md"], _link);
  main.variable(observer()).define(["md"], _27);
  main.variable(observer("theme")).define("theme", ["DOM"], _theme);
  main.variable(observer("themeCSS")).define("themeCSS", ["html","theme"], _themeCSS);
  const child1 = runtime.module(define1);
  main.import("signature", child1);
  return main;
}
