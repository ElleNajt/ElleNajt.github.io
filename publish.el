(require 'ox-publish)

(defvar elle/repo-root (file-name-directory (or load-file-name buffer-file-name))
  "Root of the repository, determined from the location of this file.")

(defvar elle/org-setup-file "templates/webpage_headers.org"
  "Path to the setup file, relative to the repository root.")

(defun elle/org-export-setup (backend)
  "Insert SETUPFILE, navigation links, and CSS link with correct relative paths.
Calculates depth from repo root so paths work for pages in any subdirectory."
  (save-excursion
    (let* ((current-file (buffer-file-name))
           (setup-file-path (expand-file-name elle/org-setup-file elle/repo-root))
           (relative-path (file-relative-name setup-file-path (file-name-directory current-file)))
           (depth (- (length (split-string (file-relative-name current-file elle/repo-root) "/")) 1))
           (prefix (if (= depth 0) "./" (string-join (make-list depth "../"))))
           (link-home (concat prefix "index.html"))
           (is-index (string= (file-name-nondirectory current-file) "index.org"))
           (dir-has-index (and (not is-index)
                               (file-exists-p (expand-file-name "index.org" (file-name-directory current-file)))))
           (link-up (cond
                     (is-index (concat prefix "index.html"))
                     (dir-has-index "./index.html")
                     (t (concat prefix "index.html"))))
           (css-path (concat prefix "css/stylesheet.css"))
           (js-path (concat prefix "scripts/org-fold.js")))
      (goto-char (point-min))
      (insert "#+SETUPFILE: " relative-path "\n")
      (insert "#+HTML_LINK_UP: " link-up "\n")
      (insert "#+HTML_LINK_HOME: " link-home "\n")
      (insert "#+HTML_HEAD: <link rel=\"stylesheet\" type=\"text/css\" href=\"" css-path "\"/>\n")
      (insert "#+HTML_HEAD: <script src=\"" js-path "\" defer></script>\n"))))

(add-hook 'org-export-before-processing-hook #'elle/org-export-setup)

(setq org-html-validation-link nil)
(setq org-html-head-include-default-style nil)

(let ((base-dir (expand-file-name "" elle/repo-root))
      (pub-dir (expand-file-name "docs" elle/repo-root)))
  (setq org-publish-project-alist
        `(("personal_webpage-pages"
           :base-directory ,base-dir
           :publishing-directory ,pub-dir
           :exclude "^docs/.*\\|^ext/.*\\|todo\\.org\\|drafts/.*"
           :recursive t
           :publishing-function org-html-publish-to-html
           :html-link-use-abs-url nil
           :html-link-org-files-as-html t
           :with-toc nil
           :section-numbers nil
           :html-extension "html")
          ("personal_webpage-static"
           :base-directory ,base-dir
           :base-extension "css\\|js\\|png\\|jpg\\|gif\\|pdf\\|mp3\\|ogg\\|swf"
           :exclude "^docs/\\|^ext/"
           :publishing-directory ,pub-dir
           :recursive t
           :publishing-function org-publish-attachment)
          ("personal_webpage" :components ("personal_webpage-pages" "personal_webpage-static")))))

(setq user-mail-address "LNAJT4@gmail.com")
(setq user-full-name "Elle Najt")

