FROM ubuntu:20.04 AS builder

ARG APPDIR=/root/app
ARG NGINX_VERSION=1.19.0
ARG NGINX_USER=www-data

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get dist-upgrade -y && \
    apt-get install -y \
    build-essential \
    git \
    wget \
    ffmpeg \
    libpcre3 \
    libpcre3-dev \
    libssl-dev \
    zlib1g-dev \
    perl \
    libperl-dev \
    libgd3 \
    libgd-dev \
    libgeoip1 \
    libgeoip-dev \
    geoip-bin \
    libxml2 \
    libxml2-dev \
    libxslt1.1 \
    libxslt1-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR ${APPDIR}

RUN git clone --depth 1 https://github.com/sergey-dryabzhinsky/nginx-rtmp-module.git && \
    wget -q https://nginx.org/download/nginx-${NGINX_VERSION}.tar.gz && \
    tar -xf nginx-${NGINX_VERSION}.tar.gz && \
    rm nginx-${NGINX_VERSION}.tar.gz && \
    cd nginx-${NGINX_VERSION} && \
    ./configure \
    --add-module=${APPDIR}/nginx-rtmp-module \
    --prefix=/etc/nginx \
    --sbin-path=/usr/sbin/nginx \
    --modules-path=/usr/lib/nginx/modules \
    --conf-path=/etc/nginx/nginx.conf \
    --error-log-path=/var/log/nginx/error.log \
    --pid-path=/var/run/nginx.pid \
    --lock-path=/var/run/nginx.lock \
    --user=${NGINX_USER} \
    --group=${NGINX_USER} \
    --build=Ubuntu \
    --with-select_module \
    --with-poll_module \
    --with-threads \
    --with-file-aio \
    --with-http_ssl_module \
    --with-http_v2_module \
    --with-http_realip_module \
    --with-http_addition_module \
    --with-http_xslt_module=dynamic \
    --with-http_image_filter_module=dynamic \
    --with-http_geoip_module=dynamic \
    --with-http_sub_module \
    --with-http_dav_module \
    --with-http_flv_module \
    --with-http_mp4_module \
    --with-http_gunzip_module \
    --with-http_gzip_static_module \
    --with-http_auth_request_module \
    --with-http_random_index_module \
    --with-http_secure_link_module \
    --with-http_degradation_module \
    --with-http_slice_module \
    --with-http_stub_status_module \
    --with-http_perl_module=dynamic \
    --with-perl_modules_path=/usr/share/perl/5.26.1 \
    --with-perl=/usr/bin/perl \
    --http-log-path=/var/log/nginx/access.log \
    --http-client-body-temp-path=/var/cache/nginx/client_temp \
    --http-proxy-temp-path=/var/cache/nginx/proxy_temp \
    --http-fastcgi-temp-path=/var/cache/nginx/fastcgi_temp \
    --http-uwsgi-temp-path=/var/cache/nginx/uwsgi_temp \
    --http-scgi-temp-path=/var/cache/nginx/scgi_temp \
    --with-mail=dynamic \
    --with-mail_ssl_module \
    --with-stream=dynamic \
    --with-stream_ssl_module \
    --with-stream_realip_module \
    --with-stream_geoip_module=dynamic \
    --with-stream_ssl_preread_module \
    --with-compat \
    --with-pcre-jit \
    --with-openssl-opt=no-nextprotoneg && \
    make -j $(nproc) && \
    make install && \
    mkdir -p /var/cache/nginx

FROM ubuntu:20.04

ARG NGINX_USER=www-data

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ffmpeg \
    libpcre3 \
    libssl1.1 \
    zlib1g \
    perl \
    libgd3 \
    libgeoip1 \
    libxml2 \
    libxslt1.1 && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /var/cache/nginx /var/log/nginx /var/www/html/stream

COPY --from=builder /usr/sbin/nginx /usr/sbin/nginx
COPY --from=builder /etc/nginx /etc/nginx
COPY --from=builder /usr/lib/nginx/modules /usr/lib/nginx/modules
COPY --chown=${NGINX_USER}:${NGINX_USER} ./stream/ /var/www/html/stream/
COPY --chown=root:root ./nginx-web/nginx.conf /etc/nginx/nginx.conf

CMD ["nginx", "-g", "daemon off;"]
