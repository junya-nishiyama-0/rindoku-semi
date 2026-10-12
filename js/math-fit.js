/**
 * Shrinks wide KaTeX display equations so they fit inside their slide.
 */
const RevealMathFit = {
  id: "math-fit",

  init(deck) {
    let scheduledFrame = null;

    const fitEquations = () => {
      scheduledFrame = null;

      document
        .querySelectorAll(".reveal .slides .katex-display")
        .forEach((display) => {
          const equation = display.querySelector(":scope > .katex");

          if (!equation) return;

          // Reset first so enlarging the browser restores the original size.
          display.style.fontSize = "";
          const baseFontSize = Number.parseFloat(
            window.getComputedStyle(display).fontSize,
          );

          // KaTeX's outer elements fill the available width even when their
          // contents overflow, so compare the layout and scroll widths.
          const availableWidth = display.clientWidth;
          const equationWidth = display.scrollWidth;

          if (availableWidth > 0 && equationWidth > availableWidth) {
            const scale = (availableWidth / equationWidth) * 0.98;
            display.style.fontSize = `${baseFontSize * scale}px`;
          }
        });
    };

    const scheduleFit = () => {
      if (scheduledFrame !== null) {
        cancelAnimationFrame(scheduledFrame);
      }

      scheduledFrame = requestAnimationFrame(fitEquations);
    };

    deck.on("ready", scheduleFit);
    deck.on("slidechanged", scheduleFit);
    window.addEventListener("resize", scheduleFit);
  },
};
