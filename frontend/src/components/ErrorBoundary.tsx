import { Component, type ReactNode } from "react";

/** Keeps a page that cannot render (typically: saved settings naming something this version does not know)
 *  from taking the whole app down, and offers the way out. */
export default class ErrorBoundary extends Component<{ children: ReactNode; onReset: () => void }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) { return { error }; }
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div role="alert" className="rounded border border-red/40 bg-red-soft p-4 text-sm">
        <div className="font-medium text-red">This page could not be displayed.</div>
        <p className="mt-1">Most likely the settings saved in this browser refer to something this version does not know ({this.state.error.message}).</p>
        <button className="btn mt-3" onClick={() => { this.props.onReset(); this.setState({ error: null }); }}>Reset saved settings</button>
      </div>
    );
  }
}
