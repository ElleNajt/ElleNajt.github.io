(require 'ox-publish)

;; https://emacs.stackexchange.com/questions/7438/how-to-include-other-org-files-programmatically-ie-not-from-main-org-file
;; (setq org-html-head-include-default-style nil)
;; (setq org-html-head " ")

(defvar elle/org-setup-file "templates/webpage_headers.org"
  "Path to the setup file, relative to the Git repository root.")

(defun elle/org-export-setup (backend)
  "Insert the SETUPFILE directive with a path relative to the current file,
and HTML_LINK_UP directive with the correct number of ../ to reach index.html."
  (interactive)
  (save-excursion
    (let* ((current-file (buffer-file-name))
           (repo-root (locate-dominating-file current-file ".git"))
           (setup-file-path (expand-file-name elle/org-setup-file repo-root))
           (relative-path (file-relative-name setup-file-path (file-name-directory current-file)))
           (depth (- (length (split-string (file-relative-name current-file repo-root) "/")) 1))
           (link-up (concat (string-join  (make-list depth "../")) "index.html")))
      (goto-char (point-min))
      (insert "#+SETUPFILE: " relative-path "\n")
      (insert "#+HTML_LINK_UP: " link-up "\n")
      (insert "#+HTML_LINK_HOME: " link-up "\n"))))


(add-hook 'org-export-before-processing-hook
          'elle/org-export-setup)

(setq org-html-validation-link nil)


(setq org-publish-project-alist
      '(("personal_webpage-pages"
         :base-directory "~/org/personal_webpage"
         :publishing-directory "~/org/personal_webpage/docs"
         :exclude "^docs/.*\\|todo\\.org\\|drafts/.*"
         :recursive t
         :publishing-function org-html-publish-to-html
         ;; :html-link-home t
         ;; by default this finds the index in the folder
         ;; which isn't the home behavior I'd expect or want
         :html-link-use-abs-url nil
         :html-link-org-files-as-html t
         :with-toc nil
         :section-numbers nil
         :html-extension "html"
         ;; :auto-sitemap t
         ;; :auto-preamble t

         ;; :auto-index t
         ;; :make-index t
         ;; :html-head "<link rel=\"stylesheet\" type=\"text/css\" href=\"../css/stylesheet.css\" />"
         )
        ("personal_webpage-static"
         :base-directory "~/org/personal_webpage"
         :base-extension "css\\|js\\|png\\|jpg\\|gif\\|pdf\\|mp3\\|ogg\\|swf"

         :exclude "^docs/"
         :publishing-directory "~/org/personal_webpage/docs"
         :recursive t
         :publishing-function org-publish-attachment
         )
        ("personal_webpage" :components ("personal_webpage-pages" "personal_webpage-static"))
        )


      )
;; (setq org-html-head-include-default-style nil)
(setq user-mail-address "LNAJT4@gmail.com")
(setq user-full-name "Elle Najt")
(org-publish-all t)

