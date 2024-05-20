![](http://media.charlesleifer.com/blog/photos/sqlite-web.png)

sqlite-web 是用Python编写的基于Web的SQLite数据库浏览器。 从coleifer/sqlite-web分叉，进行了页面汉化处理

项目依赖项:

* [flask](http://flask.pocoo.org)：轻量级的 Python Web 开发框架，为 sqlite-web 提供了便捷的 HTTP 服务和路由处理。
* [peewee](http://docs.peewee-orm.com)：一个灵活小巧的 Python ORM（对象关系映射），简化了数据库操作。
* [pygments](http://pygments.org)：代码高亮库，使得 SQL 查询和其他编程内容更易于阅读。

### 安装：

```sh
$ pip install sqlite-web
```

### 使用：

```sh
$ sqlite_web /path/to/database.db
```

### 特点：

* 可以与现有的SQLite数据库一起使用，也可以用于创建新的数据库。
* 添加或删除:
  * 表
  * 列（是的，您可以删除并重命名列！）
  * 索引
* 导出数据为JSON或CSV（导出时，注意表名不能为中文）。
* 导入JSON或CSV。
* 浏览表数据。
* 插入、更新、删除记录。

### 命令行参数：

使用语法:

```console

$ sqlite_web [options] /path/to/database-file.db
```
可用参数：
* `-p`, `--port`: 默认端口 8080
* `-H`, `--host`: 默认ip 127.0.0.1
* `-d`, `--debug`: 默认debug模式 false
* `-l`, `--log-file`: 应用程序日志的文件名。
* `-x`, `--no-browser`: 在sqlite-web启动时不打开Web浏览器。
* `-P`, `--password`: 访问sqlite-web的密码。或者，密码可以存储在"SQLITE_WEB_PASSWORD"环境变量中，在这种情况下，应用程序不会提示输入密码，而是使用环境中的值。
* `-r`, `--read-only`: 以只读模式打开数据库。
* `-R`, `--rows-per-page`: 设置内容页的分页，默认为50行。
* `-Q`, `--query-rows-per-page`: 设置查询页的分页，默认为1000行。
* `-T`, `--no-truncate`: 禁用长文本值的省略号。如果使用此选项，则始终显示完整的文本值。
* `-e`, `--extension`: 加载扩展的路径或名称。要加载多个扩展，请为每个扩展指定-e [path]。
* `-f`, `--foreign-keys`: 启用外键约束pragma。
* `-u`, `--url-prefix`: 应用程序的URL前缀，例如"/sqlite-web"。
* `-c`, `--cert` and ``-k``, ``--key`` - 指定SSL证书和私钥。
* `-a`, `--ad-hoc` - 使用临时的SSL上下文运行。

### 使用docker：

sqlite-web提供了一个Dockerfile。 使用方法：

```console

$ cd docker/  # Change dirs to the dir containing Dockerfile
$ docker build -t coleifer/sqlite-web .
$ docker run -it --rm \
    -p 8080:8080 \
    -v /path/to/your-data:/data \
    -e SQLITE_DATABASE=db_filename.db \
    coleifer/sqlite-web
```
