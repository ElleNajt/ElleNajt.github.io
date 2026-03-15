// Tag filter for blog index.
// Scans for org-exported tag spans, builds a filter bar, toggles post visibility.
document.addEventListener("DOMContentLoaded", function () {
  // Find all tag spans on the page
  var tagSpans = document.querySelectorAll(".tag > span");
  if (tagSpans.length === 0) return;

  // Collect unique tag names
  var tagSet = {};
  tagSpans.forEach(function (span) {
    tagSet[span.textContent.trim()] = true;
  });
  var allTags = Object.keys(tagSet).sort();
  if (allTags.length === 0) return;

  // Find post containers (details elements or outline-container divs containing tags)
  function getPostContainer(tagSpan) {
    var el = tagSpan.closest("details") || tagSpan.closest("div[id^='outline-container-']");
    return el;
  }

  // Build filter bar
  var filterBar = document.createElement("div");
  filterBar.className = "tag-filter-bar";
  filterBar.innerHTML = "<span class='tag-filter-label'>filter:</span> ";

  var activeTag = null;

  allTags.forEach(function (tag) {
    var btn = document.createElement("button");
    btn.className = "tag-filter-btn";
    btn.textContent = ":" + tag + ":";
    btn.dataset.tag = tag;

    btn.addEventListener("click", function () {
      if (activeTag === tag) {
        // Deactivate: show all
        activeTag = null;
        filterBar.querySelectorAll(".tag-filter-btn").forEach(function (b) {
          b.classList.remove("active");
        });
        showAll();
      } else {
        // Activate this tag
        activeTag = tag;
        filterBar.querySelectorAll(".tag-filter-btn").forEach(function (b) {
          b.classList.toggle("active", b.dataset.tag === tag);
        });
        filterByTag(tag);
      }
    });

    filterBar.appendChild(btn);
  });

  // Insert filter bar before the first post
  var firstPost = getPostContainer(tagSpans[0]);
  if (firstPost && firstPost.parentNode) {
    firstPost.parentNode.insertBefore(filterBar, firstPost);
  }

  function showAll() {
    tagSpans.forEach(function (span) {
      var container = getPostContainer(span);
      if (container) container.style.display = "";
    });
  }

  function filterByTag(tag) {
    // Collect containers and whether they match
    var seen = new Set();
    tagSpans.forEach(function (span) {
      var container = getPostContainer(span);
      if (!container || seen.has(container)) return;
      seen.add(container);

      var containerTags = container.querySelectorAll(".tag > span");
      var hasTag = Array.from(containerTags).some(function (s) {
        return s.textContent.trim() === tag;
      });
      container.style.display = hasTag ? "" : "none";
    });
  }
});
