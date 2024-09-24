(require 'ox-publish)
(setq org-publish-project-alist
      '(("blog-pages"
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
         )
        ("blog-static"
         :base-directory "~/org/blog"
         :base-extension "css\\|js\\|png\\|jpg\\|gif\\|pdf\\|mp3\\|ogg\\|swf"
         :publishing-directory "~/org/blog/docs"
         :recursive t
         :publishing-function org-publish-attachment
         )
        ("blog" :components ("blog-pages" "blog-static"))
        )


      )

(org-publish-all t)
