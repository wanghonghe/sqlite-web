#!/usr/bin/env python

__version__ = '0.6.3'

import base64
import datetime
import hashlib
import logging
import math
import operator
import optparse
import os
import re
import sys
import threading
import time
import webbrowser
from collections import namedtuple, OrderedDict
from functools import wraps
from getpass import getpass
from io import TextIOWrapper
from logging.handlers import WatchedFileHandler
from werkzeug.routing import BaseConverter

# Py2k compat.
if sys.version_info[0] == 2:
    PY2 = True
    binary_types = (buffer, bytes, bytearray)
    decode_handler = 'replace'
    numeric = (int, long, float)
    unicode_type = unicode
    from StringIO import StringIO
else:
    PY2 = False
    binary_types = (bytes, bytearray)
    decode_handler = 'backslashreplace'
    numeric = (int, float)
    unicode_type = str
    from functools import reduce
    from io import StringIO

try:
    from flask import (
        Flask, abort, flash, jsonify, make_response, redirect,
        render_template, request, session, url_for)
except ImportError:
    raise RuntimeError('无法导入flask模块。 请运行命令安装模块： '
                       'pip install flask')
try:
    from markupsafe import Markup, escape
except ImportError:
    raise RuntimeError('无法导入 markupsafe 模块。 请运行命令安装模块：'
                       ' pip install markupsafe')

try:
    from pygments import formatters, highlight, lexers
except ImportError:
    import warnings
    warnings.warn('找不到pygments 模块的library函数。', ImportWarning)
    syntax_highlight = lambda data: '<pre>%s</pre>' % data
else:
    def syntax_highlight(data):
        if not data:
            return ''
        lexer = lexers.get_lexer_by_name('sql')
        formatter = formatters.HtmlFormatter(linenos=False)
        return highlight(data, lexer, formatter)

try:
    from peewee import __version__ as _pw_version
    peewee_version = tuple([int(p) for p in _pw_version.split('.')])
except ImportError:
    raise RuntimeError('无法导入 peewee 模块。请运行命令安装模块： '
                       'pip install peewee')
else:
    if peewee_version < (3, 0, 0):
        raise RuntimeError('需求：Peewee >= 3.0.0 。发现版本： %s. '
                           '请运行pip install --update命令更新： '
                           'peewee' % _pw_version)

from peewee import *
from peewee import IndexMetadata
from peewee import sqlite3
from playhouse.dataset import DataSet
from playhouse.migrate import migrate


CUR_DIR = os.path.realpath(os.path.dirname(__file__))
DEBUG = False
ROWS_PER_PAGE = 50
QUERY_ROWS_PER_PAGE = 1000
TRUNCATE_VALUES = True
SECRET_KEY = 'sqlite-database-browser-0.1.0'

app = Flask(
    __name__,
    static_folder=os.path.join(CUR_DIR, 'static'),
    template_folder=os.path.join(CUR_DIR, 'templates'))
app.config.from_object(__name__)
dataset = None
migrator = None

#
# Database metadata objects.
#

TriggerMetadata = namedtuple('TriggerMetadata', ('name', 'sql'))

ViewMetadata = namedtuple('ViewMetadata', ('name', 'sql'))

#
# Database helpers.
#

class SqliteDataSet(DataSet):
    @property
    def filename(self):
        db_file = dataset._database.database
        if db_file.startswith('file:'):
            db_file = db_file[5:]
        return os.path.realpath(db_file.rsplit('?', 1)[0])

    @property
    def is_readonly(self):
        db_file = dataset._database.database
        return db_file.endswith('?mode=ro')

    @property
    def base_name(self):
        return os.path.basename(self.filename)

    @property
    def created(self):
        stat = os.stat(self.filename)
        return datetime.datetime.fromtimestamp(stat.st_ctime)

    @property
    def modified(self):
        stat = os.stat(self.filename)
        return datetime.datetime.fromtimestamp(stat.st_mtime)

    @property
    def size_on_disk(self):
        stat = os.stat(self.filename)
        return stat.st_size

    def get_indexes(self, table):
        return dataset._database.get_indexes(table)

    def get_all_indexes(self):
        cursor = self.query(
            'SELECT name, sql FROM sqlite_master '
            'WHERE type = ? ORDER BY name',
            ('index',))
        return [IndexMetadata(row[0], row[1], None, None, None)
                for row in cursor.fetchall()]

    def get_columns(self, table):
        return dataset._database.get_columns(table)

    def get_foreign_keys(self, table):
        return dataset._database.get_foreign_keys(table)

    def get_triggers(self, table):
        cursor = self.query(
            'SELECT name, sql FROM sqlite_master '
            'WHERE type = ? AND tbl_name = ?',
            ('trigger', table))
        return [TriggerMetadata(*row) for row in cursor.fetchall()]

    def get_all_triggers(self):
        cursor = self.query(
            'SELECT name, sql FROM sqlite_master '
            'WHERE type = ? ORDER BY name',
            ('trigger',))
        return [TriggerMetadata(*row) for row in cursor.fetchall()]

    def get_table_sql(self, table):
        if not table:
            return

        cursor = dataset.query(
            'SELECT sql FROM sqlite_master '
            'WHERE tbl_name = ? AND type IN (?, ?)',
            [table, 'table', 'view'])
        res = cursor.fetchone()
        if res is not None:
            return res[0]

    def get_view(self, name):
        cursor = self.query(
            'SELECT name, sql FROM sqlite_master '
            'WHERE type = ? AND name = ?', ('view', name))
        res = cursor.fetchone()
        if res is not None:
            return ViewMetadata(*res)

    def get_all_views(self):
        cursor = self.query(
            'SELECT name, sql FROM sqlite_master '
            'WHERE type = ? ORDER BY name',
            ('view',))
        return [ViewMetadata(*row) for row in cursor.fetchall()]

    def get_virtual_tables(self):
        cursor = self.query(
            'SELECT name FROM sqlite_master '
            'WHERE type = ? AND sql LIKE ? '
            'ORDER BY name',
            ('table', 'CREATE VIRTUAL TABLE%'))
        return set([row[0] for row in cursor.fetchall()])

    def get_corollary_virtual_tables(self):
        virtual_tables = self.get_virtual_tables()
        suffixes = ['content', 'docsize', 'segdir', 'segments', 'stat']
        return set(
            '%s_%s' % (virtual_table, suffix) for suffix in suffixes
            for virtual_table in virtual_tables)

    def is_view(self, name):
        cursor = self.query(
            'SELECT name FROM sqlite_master '
            'WHERE type = ? AND name = ?', ('view', name))
        return cursor.fetchone() is not None

    def view_operations(self, name):
        cursor = self.query(
            'SELECT sql FROM sqlite_master WHERE type=? AND tbl_name=?',
            ('trigger', name))
        triggers = [t for t, in cursor.fetchall()]
        rgx = re.compile(r'CREATE\s+TRIGGER.+?\sINSTEAD\s+OF\s+'
                         '(INSERT|UPDATE|DELETE)\s', re.I)
        operations = set()
        for trigger in triggers:
            operations.update([op.lower() for op in rgx.findall(trigger)])

        return operations

class Base64Converter(BaseConverter):
    def to_python(self, value):
        return base64.urlsafe_b64decode(value).decode()

    def to_url(self, value):
        if not isinstance(value, (bytes, str)):
            value = str(value).encode('utf8')
        elif isinstance(value, str):
            value = value.encode('utf8')
        return base64.urlsafe_b64encode(value).decode()

app.url_map.converters['b64'] = Base64Converter

#
# Flask views.
#

@app.route('/')
def index():
    return render_template('index.html', sqlite=sqlite3)

@app.route('/login/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form.get('password') == app.config['PASSWORD']:
            session['authorized'] = True
            return redirect(session.get('next_url') or url_for('index'))
        flash('您输入的密码不正确。', 'danger')
        app.logger.debug('收到错误密码尝试： %s' %
                         request.remote_addr)
    return render_template('login.html')

@app.route('/logout/', methods=['GET'])
def logout():
    session.pop('authorized', None)
    return redirect(url_for('login'))

def _query_view(template, table=None):
    data = []
    data_description = error = row_count = sql = None
    ordering = None

    sql = qsql = request.values.get('sql') or ''

    if 'export_json' in request.values:
        ordering = request.values.get('export_ordering')
        export_format = 'json'
    elif 'export_csv' in request.values:
        ordering = request.values.get('export_ordering')
        export_format = 'csv'
    else:
        ordering = request.values.get('ordering')
        export_format = None

    if ordering:
        ordering = int(ordering)
        direction = 'DESC' if ordering < 0 else 'ASC'
        qsql = ('SELECT * FROM (%s) AS _ ORDER BY %d %s' %
                (sql.rstrip(' ;'), abs(ordering), direction))
    else:
        ordering = None

    if table:
        default_sql = 'SELECT * FROM "%s"' % table
        model_class = dataset[table].model_class
    else:
        default_sql = ''
        model_class = dataset._base_model

    page = page_next = page_prev = page_start = page_end = total_pages = 1
    total = -1  # Number of rows in query result.
    if qsql:
        if export_format:
            query = model_class.raw(qsql).dicts()
            return export(query, export_format, table)

        try:
            total, = dataset.query('SELECT COUNT(*) FROM (%s) as _' %
                                   qsql.rstrip('; ')).fetchone()
        except Exception as exc:
            total = -1

        # Apply pagination.
        rpp = app.config['QUERY_ROWS_PER_PAGE']
        page = request.values.get('page')
        if page and page.isdigit():
            page = max(int(page), 1)
        else:
            page = 1
        offset = (page - 1) * rpp
        page_prev, page_next = max(page - 1, 1), page + 1

        # Figure out highest page.
        if total > 0:
            total_pages = max(1, int(math.ceil(total / float(rpp))))
            page = max(min(page, total_pages), 1)
            page_next = min(page + 1, total_pages)
            page_start = offset + 1
            page_end = min(total, page_start + rpp - 1)

        if page > 1:
            qsql = ('SELECT * FROM (%s) AS _ LIMIT %d OFFSET %d' %
                    (qsql.rstrip(' ;'), rpp, offset))

        try:
            cursor = dataset.query(qsql)
        except Exception as exc:
            error = str(exc)
            app.logger.exception('用户提交的查询出错。')
        else:
            data = cursor.fetchmany(rpp)
            data_description = cursor.description
            row_count = cursor.rowcount

    return render_template(
        template,
        data=data,
        data_description=data_description,
        default_sql=default_sql,
        error=error,
        ordering=ordering,
        page=page,
        page_end=page_end,
        page_next=page_next,
        page_prev=page_prev,
        page_start=page_start,
        query_images=get_query_images(),
        row_count=row_count,
        sql=sql,
        table=table,
        table_sql=dataset.get_table_sql(table),
        total=total,
        total_pages=total_pages)

@app.route('/query/', methods=['GET', 'POST'])
def generic_query():
    return _query_view('query.html')

def require_table(fn):
    @wraps(fn)
    def inner(table, *args, **kwargs):
        if table not in dataset.tables:
            abort(404)
        return fn(table, *args, **kwargs)
    return inner

@app.route('/create-table/', methods=['POST'])
def table_create():
    table = (request.form.get('table_name') or '').strip()
    if not table:
        flash('表名称是必需的。', 'danger')
        dest = request.form.get('redirect') or url_for('index')
        if not dest.startswith('/'):
            dest = '/' + dest
        return redirect(dest)

    try:
        dataset[table]
    except Exception as exc:
        flash('Error: %s' % str(exc), 'danger')
        app.logger.exception('尝试创建表时出错。')
    return redirect(url_for('table_import', table=table))

@app.route('/<table>/')
@require_table
def table_structure(table):
    ds_table = dataset[table]
    model_class = ds_table.model_class

    return render_template(
        'table_structure.html',
        columns=dataset.get_columns(table),
        ds_table=ds_table,
        foreign_keys=dataset.get_foreign_keys(table),
        indexes=dataset.get_indexes(table),
        model_class=model_class,
        table=table,
        table_sql=dataset.get_table_sql(table),
        triggers=dataset.get_triggers(table))

def get_request_data():
    if request.method == 'POST':
        return request.form
    return request.args

@app.route('/<table>/add-column/', methods=['GET', 'POST'])
@require_table
def add_column(table):
    class JsonField(TextField):
        field_type = 'JSON'
    column_mapping = OrderedDict((
        ('TEXT', TextField),
        ('INTEGER', IntegerField),
        ('REAL', FloatField),
        ('BLOB', BlobField),
        ('JSON', JsonField),
        ('BOOL', BooleanField),
        ('DATETIME', DateTimeField),
        ('DATE', DateField),
        ('DECIMAL', DecimalField),
        ('TIME', TimeField),
        ('VARCHAR', CharField)))

    request_data = get_request_data()
    col_type = request_data.get('type')
    name = request_data.get('name', '')

    if request.method == 'POST':
        if name and col_type in column_mapping:
            try:
                migrate(
                    migrator.add_column(
                        table,
                        name,
                        column_mapping[col_type](null=True)))
            except Exception as exc:
                flash('尝试添加列时出错 "%s": %s' % (name, exc),
                      'danger')
                app.logger.exception('尝试添加列时出错。')
            else:
                flash('列 "%s" 添加成功!' % name, 'success')
                dataset.update_cache(table)
                return redirect(url_for('table_structure', table=table))
        else:
            flash('名称和列类型是必须的。', 'danger')

    return render_template(
        'add_column.html',
        col_type=col_type,
        column_mapping=column_mapping,
        name=name,
        table=table,
        table_sql=dataset.get_table_sql(table))

@app.route('/<table>/drop-column/', methods=['GET', 'POST'])
@require_table
def drop_column(table):
    request_data = get_request_data()
    name = request_data.get('name', '')
    columns = dataset.get_columns(table)
    column_names = [column.name for column in columns]

    if request.method == 'POST':
        if name in column_names:
            try:
                migrate(migrator.drop_column(table, name))
            except Exception as exc:
                flash('尝试删除列时出错 "%s": %s' % (name, exc),
                      'danger')
                app.logger.exception('尝试删除列时出错。')
            else:
                flash('列 "%s" 删除成功!' % name, 'success')
                dataset.update_cache(table)
                return redirect(url_for('table_structure', table=table))
        else:
            flash('Name is required.', 'danger')

    return render_template(
        'drop_column.html',
        columns=columns,
        column_names=column_names,
        name=name,
        table=table)

@app.route('/<table>/rename-column/', methods=['GET', 'POST'])
@require_table
def rename_column(table):
    request_data = get_request_data()
    rename = request_data.get('rename', '')
    rename_to = request_data.get('rename_to', '')

    columns = dataset.get_columns(table)
    column_names = [column.name for column in columns]

    if request.method == 'POST':
        if (rename in column_names) and (rename_to not in column_names):
            try:
                migrate(migrator.rename_column(table, rename, rename_to))
            except Exception as exc:
                flash('尝试重命名列时出错 "%s": %s' % (name, exc),
                      'danger')
                app.logger.exception('尝试重命名列时出错。')
            else:
                flash('列 "%s" 重命名成功!' % rename, 'success')
                dataset.update_cache(table)
                return redirect(url_for('table_structure', table=table))
        else:
            flash('列名是必需的，并且不能与“现有列”的名称冲突。', 'danger')

    return render_template(
        'rename_column.html',
        columns=columns,
        column_names=column_names,
        rename=rename,
        rename_to=rename_to,
        table=table)

@app.route('/<table>/add-index/', methods=['GET', 'POST'])
@require_table
def add_index(table):
    request_data = get_request_data()
    indexed_columns = request_data.getlist('indexed_columns')
    unique = bool(request_data.get('unique'))

    columns = dataset.get_columns(table)

    if request.method == 'POST':
        if indexed_columns:
            try:
                migrate(
                    migrator.add_index(
                        table,
                        indexed_columns,
                        unique))
            except Exception as exc:
                flash('尝试创建索引时出错: %s' % exc, 'danger')
                app.logger.exception('尝试创建索引时出错。')
            else:
                flash('创建索引成功', 'success')
                return redirect(url_for('table_structure', table=table))
        else:
            flash('必须选择一个或多个列。', 'danger')

    return render_template(
        'add_index.html',
        columns=columns,
        indexed_columns=indexed_columns,
        table=table,
        unique=unique)

@app.route('/<table>/drop-index/', methods=['GET', 'POST'])
@require_table
def drop_index(table):
    request_data = get_request_data()
    name = request_data.get('name', '')
    indexes = dataset.get_indexes(table)
    index_names = [index.name for index in indexes]

    if request.method == 'POST':
        if name in index_names:
            try:
                migrate(migrator.drop_index(table, name))
            except Exception as exc:
                flash('尝试删除索引时出错: %s' % exc, 'danger')
                app.logger.exception('尝试删除索引时出错。')
            else:
                flash('索引 "%s" 删除成功!' % name, 'success')
                return redirect(url_for('table_structure', table=table))
        else:
            flash('Index name is required.', 'danger')

    return render_template(
        'drop_index.html',
        indexes=indexes,
        index_names=index_names,
        name=name,
        table=table)

@app.route('/<table>/drop-trigger/', methods=['GET', 'POST'])
@require_table
def drop_trigger(table):
    request_data = get_request_data()
    name = request_data.get('name', '')
    triggers = dataset.get_triggers(table)
    trigger_names = [trigger.name for trigger in triggers]

    if request.method == 'POST':
        if name in trigger_names:
            try:
                dataset.query('DROP TRIGGER "%s";' % name)
            except Exception as exc:
                flash('尝试删除触发器时出错: %s' % exc, 'danger')
                app.logger.exception('尝试删除触发器时出错。')
            else:
                flash('触发器 "%s" 删除成功!' % name, 'success')
                return redirect(url_for('table_structure', table=table))
        else:
            flash('Trigger name is required.', 'danger')

    return render_template(
        'drop_trigger.html',
        triggers=triggers,
        trigger_names=trigger_names,
        name=name,
        table=table)


@app.route('/<table>/content/')
@require_table
def table_content(table):
    page_number = request.args.get('page') or ''
    if page_number == 'last': page_number = '1000000'
    page_number = int(page_number) if page_number.isdigit() else 1

    dataset.update_cache(table)
    ds_table = dataset[table]
    model = ds_table.model_class

    total_rows = ds_table.all().count()
    rows_per_page = app.config['ROWS_PER_PAGE']
    total_pages = max(1, int(math.ceil(total_rows / float(rows_per_page))))
    # Restrict bounds.
    page_number = max(min(page_number, total_pages), 1)

    previous_page = page_number - 1 if page_number > 1 else None
    next_page = page_number + 1 if page_number < total_pages else None

    query = ds_table.all().paginate(page_number, rows_per_page)

    ordering = request.args.get('ordering')
    if ordering:
        field = model._meta.columns[ordering.lstrip('-')]
        if ordering.startswith('-'):
            field = field.desc()
        query = query.order_by(field)

    session['%s.last_viewed' % table] = (page_number, ordering)

    field_names = ds_table.columns
    columns = [f.column_name for f in ds_table.model_class._meta.sorted_fields]

    return render_template(
        'table_content.html',
        columns=columns,
        ds_table=ds_table,
        field_names=field_names,
        next_page=next_page,
        ordering=ordering,
        page=page_number,
        previous_page=previous_page,
        query=query,
        table=table,
        table_pk=model._meta.primary_key,
        table_sql=dataset.get_table_sql(table),
        total_pages=total_pages,
        total_rows=total_rows)

def minimal_validate_field(field, value):
    if value.lower().strip() == 'null':
        value = None
    if value is None and not field.null:
        return 'NULL', '列不允许 NULL 值。'
    if value is None:
        return None, None
    if isinstance(field, IntegerField):
        try:
            _ = int(value)
        except Exception:
            return value, '值不是一个数字。'
    elif isinstance(field, FloatField):
        try:
            _ = float(value)
        except Exception:
            return value, '值不是一个整数或实数。'
    elif isinstance(field, BooleanField):
        if value.lower() not in ('1', '0', 'true', 'false', 't', 'f'):
            return value, 'Value must be 1, 0, true, false, t or f.'
        value = True if value.lower() in ('1', 't', 'true') else False
    elif isinstance(field, BlobField):
        try:
            value = base64.b64decode(value)
        except Exception as exc:
            return value, '值必须是base64编码的二进制数据。'
    try:
        field.db_value(value)
    except Exception as exc:
        return value, str(exc)

    return value, None

@app.route('/<table>/insert/', methods=['GET', 'POST'])
@require_table
def table_insert(table):
    dataset.update_cache(table)
    model = dataset[table].model_class

    columns = []
    col_dict = {}
    row = {}
    for column in dataset.get_columns(table):
        field = model._meta.columns[column.name]
        if isinstance(field, AutoField):
            continue
        columns.append(column)
        col_dict[column.name] = column
        row[column.name] = ''

    edited = set()
    errors = {}
    if request.method == 'POST':
        insert = {}
        for key, value in request.form.items():
            if key not in col_dict: continue
            column = col_dict[key]
            edited.add(column.name)
            row[column.name] = value

            field = model._meta.columns[column.name]
            value, err = minimal_validate_field(field, value)
            if err:
                errors[key] = err
            else:
                insert[field] = value

        if errors:
            flash('一个或多个错误阻止了行的插入。',
                  'danger')
        elif insert:
            try:
                with dataset.transaction() as txn:
                    n = model.insert(insert).execute()
            except Exception as exc:
                flash('Insert failed: %s' % exc, 'danger')
                app.logger.exception('尝试将行插入时出错 %s.', table)
            else:
                flash('成功插入记录 (%s).' % n, 'success')
                return redirect(url_for(
                    'table_content',
                    table=table,
                    page='last'))
        else:
            flash('没有指定要插入的数据。', 'warning')
    else:
        edited = set(col_dict)  # Make all fields editable on load.

    return render_template(
        'table_insert.html',
        columns=columns,
        edited=edited,
        errors=errors,
        model=model,
        row=row,
        table=table)

def redirect_to_previous(table):
    page_ordering = session.get('%s.last_viewed' % table)
    if not page_ordering:
        return redirect(url_for('table_content', table=table))
    page, ordering = page_ordering
    kw = {}
    if page and page != 1:
        kw['page'] = page
    if ordering:
        kw['ordering'] = ordering
    return redirect(url_for('table_content', table=table, **kw))

@app.route('/<table>/update/<b64:pk>/', methods=['GET', 'POST'])
@require_table
def table_update(table, pk):
    dataset.update_cache(table)
    model = dataset[table].model_class
    table_pk = model._meta.primary_key
    if not table_pk:
        flash('表必须具有主键才能执行更新。', 'danger')
        return redirect(url_for('table_content', table=table))
    elif pk == '__uneditable__':
        flash('无法对主键进行编码以执行更新。', 'danger')
        return redirect(url_for('table_content', table=table))

    expr = decode_pk(model, pk)
    try:
        obj = model.get(expr)
    except model.DoesNotExist:
        pk_repr = pk_display(table_pk, pk)
        flash('无法获取具有主键的行 %s.' % str(pk_repr), 'danger')
        return redirect(url_for('table_content', table=table))

    columns = dataset.get_columns(table)
    col_dict = {}
    row = {}
    for column in columns:
        value = getattr(obj, column.name)
        if value is None:
            row[column.name] = None
        elif column.data_type.lower() == 'blob':
            row[column.name] = base64.b64encode(value).decode('utf8')
        else:
            row[column.name] = value

        col_dict[column.name] = column

    edited = set()
    errors = {}
    if request.method == 'POST':
        update = {}
        for key, value in request.form.items():
            if key not in col_dict: continue
            column = col_dict[key]
            edited.add(column.name)
            row[column.name] = value

            field = model._meta.columns[column.name]
            value, err = minimal_validate_field(field, value)
            if err:
                errors[key] = err
            else:
                update[field] = value

        if errors:
            flash('一个或多个错误阻止了行的更新。',
                  'danger')
        elif update:
            try:
                with dataset.transaction() as txn:
                    n = model.update(update).where(expr).execute()
            except Exception as exc:
                flash('更新失败: %s' % exc, 'danger')
                app.logger.exception('尝试从 %s 更新行时出错。', table)
            else:
                flash('成功更新 %s 记录。' % n, 'success')
                return redirect_to_previous(table)
        else:
            flash('没有指定要更新的数据。', 'warning')

    return render_template(
        'table_update.html',
        columns=columns,
        edited=edited,
        errors=errors,
        model=model,
        pk=pk,
        row=row,
        table=table,
        table_pk=model._meta.primary_key)

@app.route('/<table>/delete/<b64:pk>/', methods=['GET', 'POST'])
@require_table
def table_delete(table, pk):
    dataset.update_cache(table)
    model = dataset[table].model_class
    table_pk = model._meta.primary_key
    if not table_pk:
        flash('表必须具有主键才能执行删除。', 'danger')
        return redirect(url_for('table_content', table=table))
    elif pk == '__uneditable__':
        flash('无法对主键进行编码以执行删除。', 'danger')
        return redirect(url_for('table_content', table=table))

    expr = decode_pk(model, pk)
    try:
        row = model.select().where(expr).dicts().get()
    except model.DoesNotExist:
        pk_repr = pk_display(table_pk, pk)
        flash('无法获取具有主键的行 %s.' % str(pk_repr), 'danger')
        return redirect(url_for('table_content', table=table))

    if request.method == 'POST':
        try:
            with dataset.transaction() as txn:
                n = model.delete().where(expr).execute()
        except Exception as exc:
            flash('删除失败: %s' % exc, 'danger')
            app.logger.exception('尝试从 %s 中删除行时出错.', table)
        else:
            flash('成功删除 %s 记录.' % n, 'success')
            return redirect_to_previous(table)

    return render_template(
        'table_delete.html',
        column_names=[c.name for c in dataset.get_columns(table)],
        model=model,
        pk=pk,
        row=row,
        table=table,
        table_pk=table_pk)

@app.route('/<table>/query/', methods=['GET', 'POST'])
@require_table
def table_query(table):
    return _query_view('table_query.html', table)

def export(query, export_format, table=None):
    buf = StringIO()
    if export_format == 'json':
        kwargs = {'indent': 2}
        filename = 'export.json'
        mimetype = 'application/json'
    else:
        kwargs = {}
        filename = 'export.csv'
        mimetype = 'text/csv'

    if table:
        filename = '%s-%s' % (table, filename)

    # Avoid any special chars in export filename.
    filename = re.sub(r'[^\w\d\-\.]+', '', filename)

    dataset.freeze(query, export_format, file_obj=buf, **kwargs)

    response_data = buf.getvalue()
    response = make_response(response_data)
    response.headers['Content-Length'] = len(response_data)
    response.headers['Content-Type'] = mimetype
    response.headers['Content-Disposition'] = 'attachment; filename="%s"' % (
        filename)
    response.headers['Expires'] = 0
    response.headers['Pragma'] = 'public'
    return response

@app.route('/<table>/export/', methods=['GET', 'POST'])
@require_table
def table_export(table):
    columns = dataset.get_columns(table)
    if request.method == 'POST':
        export_format = request.form.get('export_format') or 'json'
        col_dict = {c.name: c for c in columns}
        selected = [c for c in (request.form.getlist('columns') or [])
                    if c in col_dict]
        if not selected:
            flash('请选择一个或多个要导出的列。', 'danger')
        else:
            model = dataset[table].model_class
            fields = [model._meta.columns[c] for c in selected]
            query = model.select(*fields).dicts()
            try:
                return export(query, export_format, table)
            except Exception as exc:
                flash('生成导出时出错: %s' % exc, 'danger')
                app.logger.exception('生成导出时出错。')

    return render_template(
        'table_export.html',
        columns=columns,
        table=table)

@app.route('/<table>/import/', methods=['GET', 'POST'])
@require_table
def table_import(table):
    count = None
    request_data = get_request_data()
    strict = bool(request_data.get('strict'))

    if request.method == 'POST':
        file_obj = request.files.get('file')
        if not file_obj:
            flash('请选择导入文件。', 'danger')
        elif not file_obj.filename.lower().endswith(('.csv', '.json')):
            flash('不支持此文件类型。 必须是 .json 或 .csv 文件。',
                  'danger')
        else:
            if file_obj.filename.lower().endswith('.json'):
                format = 'json'
            else:
                format = 'csv'

            # Here we need to translate the file stream. Werkzeug uses a
            # spooled temporary file opened in wb+ mode, which is not
            # compatible with Python's CSV module. We'd need to reach pretty
            # far into Flask's internals to modify this behavior, so instead
            # we'll just translate the stream into utf8-decoded unicode.
            if not PY2:
                try:
                    stream = TextIOWrapper(file_obj, encoding='utf8')
                except AttributeError:
                    # The SpooledTemporaryFile used by werkzeug does not
                    # implement an API that the TextIOWrapper expects, so we'll
                    # just consume the whole damn thing and decode it.
                    # Fixed in werkzeug 0.15.
                    stream = StringIO(file_obj.read().decode('utf8'))
            else:
                stream = file_obj.stream

            try:
                with dataset.transaction():
                    count = dataset.thaw(
                        table,
                        format=format,
                        file_obj=stream,
                        strict=strict)
            except Exception as exc:
                flash('导入文件错误: %s' % exc, 'danger')
                app.logger.exception('导入文件错误。')
            else:
                flash(
                    '已成功从 %s 导入 %s 对象。' % (
                        count, file_obj.filename),
                    'success')
                return redirect(url_for('table_content', table=table))

    return render_template(
        'table_import.html',
        count=count,
        strict=strict,
        table=table)

@app.route('/<table>/drop/', methods=['GET', 'POST'])
@require_table
def drop_table(table):
    is_view = any(v.name == table for v in dataset.get_all_views())
    label = 'view' if is_view else 'table'
    if request.method == 'POST':
        try:
            if is_view:
                dataset.query('DROP VIEW "%s";' % table)
            else:
                model_class = dataset[table].model_class
                model_class.drop_table()
        except Exception as exc:
            flash('尝试删除时出错 %s "%s".' % (label, table), 'danger')
            app.logger.exception('尝试删除时出错 %s "%s".', label, table)
        else:
            dataset.update_cache()  # Update all tables.
            flash('%s "%s" 删除成功。' %
                  ('view' if is_view else 'table', table),
                  'success')
            return redirect(url_for('index'))

    return render_template('drop_table.html', is_view=is_view, table=table)

@app.template_filter('format_index')
def format_index(index_sql):
    split_regex = re.compile(r'\bon\b', re.I)
    if not split_regex.search(index_sql):
        return index_sql

    create, definition = split_regex.split(index_sql)
    return '\nON '.join((create.strip(), definition.strip()))

@app.template_filter('encode_pk')
def encode_pk(row, pk):
    if isinstance(pk, CompositeKey):
        try:
            return ':::'.join([str(row[k]) for k in pk.field_names])
        except Exception as exc:
            return '__uneditable__'
    return row[pk.column_name]

def decode_pk(model, pk_data):
    pk = model._meta.primary_key
    if isinstance(pk, CompositeKey):
        fields = [pk.model._meta.columns[f] for f in pk.field_names]
        values = pk_data.split(':::')
        expressions = [(f == v) for f, v in zip(fields, values)]
        return reduce(operator.and_, expressions)
    return (pk == pk_data)

@app.template_filter('pk_display')
def pk_display(table_pk, pk):
    if isinstance(table_pk, CompositeKey):
        return tuple(pk.split(':::'))
    return pk

@app.template_filter('value_filter')
def value_filter(value, max_length=50):
    if isinstance(value, numeric):
        return value

    if isinstance(value, binary_types):
        if not isinstance(value, (bytes, bytearray)):
            value = bytes(value)  # Handle `buffer` type.
        value = base64.b64encode(value)[:1024].decode('utf8')
    if isinstance(value, unicode_type):
        value = escape(value)
        if len(value) > max_length and app.config['TRUNCATE_VALUES']:
            return ('<span class="truncated">%s</span> '
                    '<span class="full" style="display:none;">%s</span>'
                    '<a class="toggle-value" href="#">...</a>') % (
                        value[:max_length],
                        value)
    return value

column_re = re.compile('(.+?)\((.+)\)', re.S)
column_split_re = re.compile(r'(?:[^,(]|\([^)]*\))+')

def _format_create_table(sql):
    create_table, column_list = column_re.search(sql).groups()
    columns = ['  %s' % column.strip()
               for column in column_split_re.findall(column_list)
               if column.strip()]
    return '%s (\n%s\n)' % (
        create_table,
        ',\n'.join(columns))

@app.template_filter()
def format_create_table(sql):
    try:
        return _format_create_table(sql)
    except:
        return sql

@app.template_filter('highlight')
def highlight_filter(data):
    return Markup(syntax_highlight(data))

def get_query_images():
    accum = []
    image_dir = os.path.join(app.static_folder, 'img')
    if not os.path.exists(image_dir):
        return accum
    for filename in sorted(os.listdir(image_dir)):
        basename = os.path.splitext(os.path.basename(filename))[0]
        parts = basename.split('-')
        accum.append((parts, 'img/' + filename))
    return accum

#
# Flask application helpers.
#

@app.context_processor
def _general():
    return {
        'dataset': dataset,
        'login_required': bool(app.config.get('PASSWORD')),
        'version': __version__,
    }

@app.context_processor
def _now():
    return {'now': datetime.datetime.now()}

@app.before_request
def _connect_db():
    dataset.connect()

@app.teardown_request
def _close_db(exc):
    if not dataset._database.is_closed():
        dataset.close()


class PrefixMiddleware(object):
    def __init__(self, app, prefix):
        self.app = app
        self.prefix = '/%s' % prefix.strip('/')
        self.prefix_len = len(self.prefix)

    def __call__(self, environ, start_response):
        if environ['PATH_INFO'].startswith(self.prefix):
            environ['PATH_INFO'] = environ['PATH_INFO'][self.prefix_len:]
            environ['SCRIPT_NAME'] = self.prefix
            return self.app(environ, start_response)
        else:
            start_response('404', [('Content-Type', 'text/plain')])
            return ['URL与应用程序前缀不匹配。'.encode()]

#
# Script options.
#

def get_option_parser():
    parser = optparse.OptionParser()
    parser.add_option(
        '-p',
        '--port',
        default=8080,
        help='web端口, default=8080',
        type='int')
    parser.add_option(
        '-H',
        '--host',
        default='127.0.0.1',
        help='访问地址, default=127.0.0.1')
    parser.add_option(
        '-d',
        '--debug',
        action='store_true',
        help='在调试模式下运行服务器')
    parser.add_option(
        '-x',
        '--no-browser',
        action='store_false',
        default=True,
        dest='browser',
        help='不要自动打开浏览器页面。')
    parser.add_option(
        '-l',
        '--log-file',
        dest='log_file',
        help='程序log文件名。')
    parser.add_option(
        '-P',
        '--password',
        action='store_true',
        dest='prompt_password',
        help='提示输入密码以访问数据库浏览器。')
    parser.add_option(
        '-r',
        '--read-only',
        action='store_true',
        dest='read_only',
        help='以只读模式打开数据库。')
    parser.add_option(
        '-R',
        '--rows-per-page',
        default=50,
        dest='rows_per_page',
        help='在内容选项卡中每页显示的行数 (default=50)',
        type='int')
    parser.add_option(
        '-Q',
        '--query-rows-per-page',
        default=1000,
        dest='query_rows_per_page',
        help='在查询选项卡中每页显示的行数 (default=1000)',
        type='int')
    parser.add_option(
        '-T',
        '--no-truncate',
        action='store_false',
        default=True,
        dest='truncate_values',
        help='禁用截断长文本值。默认情况下，文本值在50个字符后省略号，单击时显示全文。')
    parser.add_option(
        '-u',
        '--url-prefix',
        dest='url_prefix',
        help='应用程序的URL前缀。')
    parser.add_option(
        '-f',
        '--foreign-keys',
        action='store_true',
        dest='foreign_keys',
        help='启用foreign_keys杂注。')
    parser.add_option(
        '-e',
        '--extension',
        action='append',
        dest='extensions',
        help='可加载扩展的路径或名称。')
    ssl_opts = optparse.OptionGroup(parser, 'SSL options')
    ssl_opts.add_option(
        '-c',
        '--ssl-cert',
        dest='ssl_cert',
        help='SSL 证书文件路径。')
    ssl_opts.add_option(
        '-k',
        '--ssl-key',
        dest='ssl_key',
        help='SSL 私有 key 文件路径。')
    ssl_opts.add_option(
        '-a',
        '--ad-hoc',
        action='store_true',
        dest='ssl_ad_hoc',
        help='Use ad-hoc SSL context.')
    parser.add_option_group(ssl_opts)
    return parser

def die(msg, exit_code=1):
    sys.stderr.write('%s\n' % msg)
    sys.stderr.flush()
    sys.exit(exit_code)

def open_browser_tab(host, port):
    url = 'http://%s:%s/' % (host, port)

    def _open_tab(url):
        time.sleep(1.5)
        webbrowser.open_new_tab(url)

    thread = threading.Thread(target=_open_tab, args=(url,))
    thread.daemon = True
    thread.start()

def install_auth_handler(password):
    app.config['PASSWORD'] = password

    @app.before_request
    def check_password():
        if not session.get('authorized') and request.path != '/login/' and \
           not request.path.startswith(('/static/', '/favicon')):
            flash('您必须登录才能查看数据库浏览器。', 'danger')
            session['next_url'] = request.base_url
            return redirect(url_for('login'))

def initialize_app(filename, read_only=False, password=None, url_prefix=None,
                   extensions=None, foreign_keys=None):
    global dataset
    global migrator

    if password:
        install_auth_handler(password)

    dataset_kw = {}
    if peewee_version >= (3, 14, 9):
        dataset_kw['include_views'] = True

    if read_only:
        if sys.version_info < (3, 4, 0):
            die('只读访问需要Python 3.4.0或更新版本。')
        if peewee_version < (3, 5, 1):
            die('只读访问需要Peewee 3.5.1或更新版本。')
        db = SqliteDatabase('file:%s?mode=ro' % filename, uri=True)
        try:
            db.connect()
        except OperationalError:
            die('无法以只读模式打开数据库文件。确保数据库存在，以便使用只读模式。')
        db.close()
    else:
        db = SqliteDatabase(filename)

    if foreign_keys:
        db.pragma('foreign_keys', True, permanent=True)

    if extensions:
        # Load extensions before performing introspection.
        for ext in extensions:
            db.load_extension(ext)

    dataset = SqliteDataSet(db, bare_fields=True, **dataset_kw)

    if url_prefix:
        app.wsgi_app = PrefixMiddleware(app.wsgi_app, prefix=url_prefix)

    migrator = dataset._migrator
    dataset.close()

def main():
    # This function exists to act as a console script entry-point.
    parser = get_option_parser()
    options, args = parser.parse_args()
    if not args:
        die('错误：缺少数据库文件所需的路径。')

    if options.log_file:
        fmt = logging.Formatter('[%(asctime)s] - [%(levelname)s] - %(message)s')
        handler = WatchedFileHandler(options.log_file)
        handler.setLevel(logging.DEBUG if options.debug else logging.WARNING)
        handler.setFormatter(fmt)
        app.logger.addHandler(handler)

        # Remove default handler.
        from flask.logging import default_handler
        app.logger.removeHandler(default_handler)

    password = None
    if options.prompt_password:
        if os.environ.get('SQLITE_WEB_PASSWORD'):
            password = os.environ['SQLITE_WEB_PASSWORD']
        else:
            while True:
                password = getpass('输入密码: ')
                password_confirm = getpass('确认密码: ')
                if password != password_confirm:
                    print('密码不匹配！')
                else:
                    break

    if options.rows_per_page:
        app.config['ROWS_PER_PAGE'] = options.rows_per_page
    if options.query_rows_per_page:
        app.config['QUERY_ROWS_PER_PAGE'] = options.query_rows_per_page

    app.config['TRUNCATE_VALUES'] = options.truncate_values

    # Initialize the dataset instance and (optionally) authentication handler.
    initialize_app(args[0], options.read_only, password, options.url_prefix,
                   options.extensions, options.foreign_keys)

    if options.browser:
        open_browser_tab(options.host, options.port)

    if password:
        key = b'sqlite-web-' + args[0].encode('utf8') + password.encode('utf8')
        app.secret_key = hashlib.sha256(key).hexdigest()

    # Set up SSL context, if specified.
    kwargs = {}
    if options.ssl_ad_hoc:
        kwargs['ssl_context'] = 'adhoc'

    if options.ssl_cert and options.ssl_key:
        if not os.path.exists(options.ssl_cert) or not os.path.exists(options.ssl_key):
            die('找不到ssl证书或ssl密钥。请检查文件路径。')
        kwargs['ssl_context'] = (options.ssl_cert, options.ssl_key)
    elif options.ssl_cert:
        die('ssl证书的“-k”需要ssl证书')
    elif options.ssl_key:
        die('ssl证书的"-c"需要ssl证书')

    # Run WSGI application.
    app.run(host=options.host, port=options.port, debug=options.debug, **kwargs)


if __name__ == '__main__':
    main()
