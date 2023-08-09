tmux new -s session_name
tmux attach -t session_name
關閉當前視窗（窗格）：在 tmux 會話內，按下 Ctrl+b（先放開），然後再按下 x。這將會提示你確認是否要關閉當前視窗，輸入 y 來確認。
關閉會話：在 tmux 會話內，按下 Ctrl+b（先放開），然後再按下 :（冒號） 進入命令模式。在命令提示符下，輸入 kill-session 或 kill-session -t session_name（如果你指定了會話名稱），然後按下 Enter 鍵。這將結束整個 tmux 會話。

git add --all
git commit -m "message"
git push

zip -r myZippedImages.zip images_folder

- git lfs
https://xuzhougeng.top/archives/upload-large-file-to-github