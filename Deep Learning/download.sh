mkdir data

wget -qO ./data/original.tar.gz https://www.dropbox.com/s/0vv2qsc4ywb4z5v/original.tar.gz?dl=0
wget -qO ./data/mixed_next.tar.gz https://www.dropbox.com/s/4hnkbvxastpcgz2/mixed_next.tar.gz?dl=0
wget -qO ./data/mixed_rand.tar.gz https://www.dropbox.com/s/cto15ceadgraur2/mixed_rand.tar.gz?dl=0
wget -qO ./data/mixed_same.tar.gz https://www.dropbox.com/s/f2525w5aqq67kk0/mixed_same.tar.gz?dl=0
wget -qO ./data/only_fg.tar.gz https://www.dropbox.com/s/alrf3jo8yyxzyrn/only_fg.tar.gz?dl=0
wget -qO ./data/only_bg_t.tar.gz https://www.dropbox.com/s/03lk878q73hyjpi/only_bg_t.tar.gz?dl=0
wget -qO ./data/only_bg_b.tar.gz https://www.dropbox.com/s/u1iekdnwail1d9u/only_bg_b.tar.gz?dl=0
wget -qO ./data/no_fg.tar.gz https://www.dropbox.com/s/0v6w9k7q7i1ytvr/no_fg.tar.gz?dl=0
wget -qO ./data/in9l.tar.gz https://www.dropbox.com/s/8w29bg9niya19rn/in9l.tar.gz?dl=0


tar -xzf ./data/original.tar.gz -C ./data
tar -xzf ./data/mixed_next.tar.gz -C ./data
tar -xzf ./data/mixed_rand.tar.gz -C ./data
tar -xzf ./data/mixed_same.tar.gz -C ./data
tar -xzf ./data/only_fg.tar.gz -C ./data
tar -xzf ./data/only_bg_t.tar.gz -C ./data
tar -xzf ./data/only_bg_b.tar.gz -C ./data
tar -xzf ./data/no_fg.tar.gz -C ./data
tar -xzf ./data/in9l.tar.gz -C ./data
