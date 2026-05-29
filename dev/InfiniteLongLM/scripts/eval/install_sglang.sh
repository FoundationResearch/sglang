source /root/sglang/python/.venv/bin/activate
git config --global --add safe.directory /apdcephfs_fsgm/share_303843174/user/guhao/SGLang-HSA
pip install --no-deps -e /apdcephfs_fsgm/share_303843174/user/guhao/SGLang-HSA/python/

source /apdcephfs_tj5/share_300719894/user/guhao/SGLang-HSA/.sglang/bin/activate

python -m pip install --verbose git+ssh://git@github.com/open-compass/human-eval.git

python -m pip install --verbose 'git+https://github.com/open-compass/human-eval.git#egg=human-eval&subdirectory=.' 
python -m pip install --verbose 'git+https://github.com/open-compass/human-eval.git#egg=evalplus&subdirectory=evalplus'


python -m pip install -e /data/home/marcushaogu/human-eval

pip install evalplus --upgrade