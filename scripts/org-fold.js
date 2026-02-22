// Convert org-mode outline sections into collapsible <details>/<summary> elements.
// Runs on DOMContentLoaded so no framework needed.
document.addEventListener("DOMContentLoaded", function () {
  // Match all outline container divs (outline-2, outline-3, etc.)
  var containers = document.querySelectorAll('[class^="outline-"][class$="]"]');
  // That selector is fragile; use a broader match:
  document.querySelectorAll("div[id^='outline-container-']").forEach(function (container) {
    var heading = container.querySelector("h2, h3, h4, h5, h6");
    var content = container.querySelector("div[class^='outline-text-']");
    if (!heading) return;

    // Collect all sibling content: the outline-text div plus any nested outline containers
    var children = Array.from(container.children).filter(function (el) {
      return el !== heading;
    });

    // Skip if there's no content below the heading
    if (children.length === 0) return;

    var details = document.createElement("details");
    details.className = container.className;
    details.id = container.id;
    details.open = true; // Start expanded like org default

    var summary = document.createElement("summary");
    // Move the heading's content into the summary
    summary.innerHTML = heading.innerHTML;
    // Copy heading tag name as a data attribute for CSS styling
    summary.dataset.level = heading.tagName.toLowerCase();
    summary.className = "org-heading " + heading.tagName.toLowerCase();

    details.appendChild(summary);
    children.forEach(function (child) {
      details.appendChild(child);
    });

    container.parentNode.replaceChild(details, container);
  });
});
