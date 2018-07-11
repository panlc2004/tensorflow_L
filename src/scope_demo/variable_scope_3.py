import tensorflow as tf

with tf.variable_scope("foo") as foo_scope:
    assert foo_scope.name == "foo"
with tf.variable_scope("bar"):
    with tf.variable_scope("baz") as other_scope:
        assert other_scope.name == "bar/baz"
        with tf.variable_scope(foo_scope) as foo_scope2:
            assert foo_scope2.name == "foo"  # Not changed.

with tf.variable_scope("test1"):
    with tf.name_scope("test2"):
        a = tf.Variable([1.0], dtype=tf.float32)
print(a.name)

with tf.variable_scope("test1"):
    with tf.name_scope("test2"):
        a = tf.Variable([1.0], dtype=tf.float32)
print(a.name)

with tf.name_scope("test1"):
    with tf.variable_scope("test2"):
        a = tf.Variable([1.0], dtype=tf.float32)
print(a.name)