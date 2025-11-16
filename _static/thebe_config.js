// Thebe Configuration for ACME Tutorials
// This overrides the default configuration to ensure Binder connection works properly

window.thebe_config = {
  requestKernel: true,
  binderOptions: {
    repo: "samueleronchini/acme_tutorials",
    ref: "main",
    binderUrl: "https://mybinder.org",
    repoProvider: "github"
  },
  kernelOptions: {
    name: "python3",
    kernelName: "python3"
  },
  selector: "div.thebelab-cell",
  mountActivateWidget: true,
  mountStatusWidget: true,
  bootstrap: true
};

console.log("✅ Thebe config loaded:", window.thebe_config);
