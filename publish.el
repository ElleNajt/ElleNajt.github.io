(require 'ox-publish)
(setq org-publish-project-alist
      '(("my_blog"
         :base-directory "~/org/blog"
         :publishing-directory "~/org/blog/docs"
         :recursive t
         :publishing-function org-html-publish-to-html
         ;; :html-link-home "index.html"
         ;; by default this finds the index in the folder
         ;; which isn't the home behavior I'd expect or want
         :html-link-use-abs-url nil
         :html-link-org-files-as-html t
         :with-toc nil
         :section-numbers nil
         :html-extension "html"
         ;; :auto-preamble t
         )))

(org-link-set-parameters
 "orgfile"
 :follow (lambda (path) (find-file path))
 :export (lambda (path desc backend)
           (cond
            ((eq backend 'html)
             (format "<a href=\"%s.org\">%s</a>" path (or desc path)))
            (t
             (format "[[file:%s.org][%s]]" path (or desc path))))))

(org-publish-all t)
