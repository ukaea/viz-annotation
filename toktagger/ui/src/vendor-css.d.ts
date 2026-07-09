// Side-effect CSS imports from dependencies.

declare module "@annotorious/react/annotorious-react.css";
declare module "react-contexify/ReactContexify.css";

declare module "*.svelte" {
  import type { Component } from "svelte";

  const component: Component<Record<string, unknown>>;
  export default component;
}
