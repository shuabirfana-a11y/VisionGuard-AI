// Kept external so printing works with script-src 'self' (no inline handlers).
document.querySelector("#print-report")?.addEventListener("click", () => window.print());
