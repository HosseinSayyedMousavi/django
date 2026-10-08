import datetime
from unittest import mock

from django.db import NotSupportedError, connections
from django.db.models import F
from django.db.models.functions import Lower
from django.db.models.sql.compiler import cursor_iter
from django.test import TestCase

from .models import Article, Note, Tag


class QuerySetIteratorTests(TestCase):
    itersize_index_in_mock_args = 3

    @classmethod
    def setUpTestData(cls):
        Article.objects.create(name="Article 1", created=datetime.datetime.now())
        Article.objects.create(name="Article 2", created=datetime.datetime.now())

    def test_iterator_invalid_chunk_size(self):
        for size in (0, -1):
            with self.subTest(size=size):
                with self.assertRaisesMessage(
                    ValueError, "Chunk size must be strictly positive."
                ):
                    Article.objects.iterator(chunk_size=size)

    def test_default_iterator_chunk_size(self):
        qs = Article.objects.iterator()
        with mock.patch(
            "django.db.models.sql.compiler.cursor_iter", side_effect=cursor_iter
        ) as cursor_iter_mock:
            next(qs)
        self.assertEqual(cursor_iter_mock.call_count, 1)
        mock_args, _mock_kwargs = cursor_iter_mock.call_args
        self.assertEqual(mock_args[self.itersize_index_in_mock_args], 2000)

    def test_iterator_chunk_size(self):
        batch_size = 3
        qs = Article.objects.iterator(chunk_size=batch_size)
        with mock.patch(
            "django.db.models.sql.compiler.cursor_iter", side_effect=cursor_iter
        ) as cursor_iter_mock:
            next(qs)
        self.assertEqual(cursor_iter_mock.call_count, 1)
        mock_args, _mock_kwargs = cursor_iter_mock.call_args
        self.assertEqual(mock_args[self.itersize_index_in_mock_args], batch_size)

    def test_keyset_iterator(self):
        now = datetime.datetime.now()
        for i, name in enumerate(["b", "a", "b", "a", "c"]):
            Article.objects.create(
                name=name, created=now + datetime.timedelta(seconds=i)
            )
        orderings = [
            ("pk",),
            ("-pk",),
            ("name",),
            ("-name",),
            ("name", "-created"),
            ("-name", "created"),
            ("created", "-pk"),
            (F("name").desc(), "created"),
            ("name", "name", "pk"),
        ]
        total = Article.objects.count()
        for ordering in orderings:
            qs = Article.objects.order_by(*ordering)
            # Ties are broken by pk, as keyset_iterator() does.
            expected = list(qs.order_by(*ordering, "pk"))
            for chunk_size in (1, 2, total, total + 1):
                with self.subTest(ordering=ordering, chunk_size=chunk_size):
                    self.assertEqual(list(qs.keyset_iterator(chunk_size)), expected)

    def test_keyset_iterator_default_ordering(self):
        for name in ["b", "a", "c"]:
            Tag.objects.create(name=name)
        expected = list(Tag.objects.all())
        self.assertEqual([t.name for t in expected], ["a", "b", "c"])
        self.assertEqual(list(Tag.objects.keyset_iterator(2)), expected)

    def test_keyset_iterator_queries(self):
        qs = Article.objects.all()
        # 2 rows with chunk_size=1 need 3 queries: the last one is empty.
        with self.assertNumQueries(3):
            list(qs.keyset_iterator(1))
        # A full last chunk requires one more query to detect the end.
        with self.assertNumQueries(2):
            list(qs.keyset_iterator(2))
        # A short last chunk ends the iteration without an extra query.
        with self.assertNumQueries(1):
            list(qs.keyset_iterator(3))

    def test_keyset_iterator_filtered(self):
        article = Article.objects.get(name="Article 2")
        qs = Article.objects.filter(name="Article 2")
        self.assertEqual(list(qs.keyset_iterator(1)), [article])

    def test_keyset_iterator_empty(self):
        self.assertEqual(list(Article.objects.none().keyset_iterator(1)), [])

    def test_keyset_iterator_prefetch_related(self):
        tag = Tag.objects.create(name="tag")
        note = Note.objects.create(note="note", misc="misc", tag=tag)
        Note.objects.create(note="note 2", misc="misc", tag=tag)
        with self.assertNumQueries(3):
            results = list(Tag.objects.prefetch_related("note_set").keyset_iterator(1))
        with self.assertNumQueries(0):
            self.assertEqual(len(results[0].note_set.all()), 2)
        self.assertIn(note, results[0].note_set.all())

    def test_keyset_iterator_invalid_chunk_size(self):
        for size in (0, -1):
            with self.subTest(size=size):
                with self.assertRaisesMessage(
                    ValueError, "Chunk size must be strictly positive."
                ):
                    Article.objects.keyset_iterator(size)

    def test_keyset_iterator_unsupported_queries(self):
        tests = [
            (
                Article.objects.values("name"),
                TypeError,
                "Cannot call keyset_iterator() after .values() or .values_list().",
            ),
            (
                Article.objects.all()[:1],
                TypeError,
                "Cannot call keyset_iterator() once a slice has been taken.",
            ),
            (
                Article.objects.distinct("name"),
                TypeError,
                "Cannot call keyset_iterator() after .distinct(*fields).",
            ),
            (
                Article.objects.union(Article.objects.all()),
                NotSupportedError,
                "Calling QuerySet.keyset_iterator() after union() is not supported.",
            ),
        ]
        for qs, exception, msg in tests:
            with self.subTest(msg=msg):
                with self.assertRaisesMessage(exception, msg):
                    qs.keyset_iterator(1)

    def test_keyset_iterator_unsupported_ordering(self):
        msg = "keyset_iterator() only supports ordering by non-nullable fields"
        tests = [
            Note.objects.order_by("tag_id"),
            Note.objects.order_by("tag"),
            Article.objects.order_by("?"),
            Article.objects.order_by(Lower("name")),
            Article.objects.annotate(lower_name=Lower("name")).order_by("lower_name"),
            Article.objects.order_by(F("name").asc(nulls_first=True)),
            Article.objects.extra(order_by=["name"]),
        ]
        for qs in tests:
            with self.subTest(query=str(qs.query)):
                with self.assertRaisesMessage(NotSupportedError, msg):
                    qs.keyset_iterator(1)

    def test_no_chunked_reads(self):
        """
        If the database backend doesn't support chunked reads, then the
        result of SQLCompiler.execute_sql() is a list.
        """
        qs = Article.objects.all()
        compiler = qs.query.get_compiler(using=qs.db)
        features = connections[qs.db].features
        with mock.patch.object(features, "can_use_chunked_reads", False):
            result = compiler.execute_sql(chunked_fetch=True)
        self.assertIsInstance(result, list)
