(require 'ox-publish)
(setq org-publish-project-alist
      '(("personal_webpage-pages"
         :base-directory "~/org/personal_webpage"
         :publishing-directory "~/org/personal_webpage/docs"
         :exclude "^docs/"
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
         :auto-sitemap t
         :auto-preamble t

         ;; :auto-index t
         ;; :make-index t
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

(org-publish-all t)
