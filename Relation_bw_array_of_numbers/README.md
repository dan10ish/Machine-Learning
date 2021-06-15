#Model to Predict Relation Between Array of Numbers

We have two variables xs and ys that have an **relation**.

`xs = -1, 0, 1, 2, 3, 4`

`ys = -3, -1, 1, 3, 5, 7`

The relation between them can be given as:

`ys = 2xs -1`

So if you substitute in place of xs as -1 you will get the corresponding value of ys ,i.e., -3

We will try to predict the relation `ys = 2xs -1` with minimum loss as possible.


-----------------------------------------------------------------

#Result

The model has predicted `[array([[1.9965848]], dtype=float32), array([-0.98941153], dtype=float32)]`

Which simply means `ys = 1.9965848xs -0.98941153`

This is very close to the **accurate answer** ,i.e, `ys = 2xs -1`