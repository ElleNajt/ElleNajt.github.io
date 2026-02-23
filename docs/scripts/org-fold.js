// Convert org-mode outline sections into collapsible <details>/<summary> elements.
// Runs on DOMContentLoaded so no framework needed.
document.addEventListener("DOMContentLoaded", function () {
  // Process deepest containers first so nested sections are converted
  // before their parents move them.
  var containers = Array.from(
    document.querySelectorAll("div[id^='outline-container-']")
  );
  containers.reverse();

  containers.forEach(function (container) {
    var heading = container.querySelector(":scope > h2, :scope > h3, :scope > h4, :scope > h5, :scope > h6");
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
    // Preserve the heading's id so TOC anchor links still work
    if (heading.id) {
      summary.id = heading.id;
    }
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
