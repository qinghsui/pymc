Pymc 3 is built around Theano, a computational graph package. Many of the ideas and syntax is shared, so being familiar with Theano is very useful. We recommend working through, or at least reading the Theano tutorial (http://deeplearning.net/software/theano/tutorial/adding.html).    

In PyMC random variables, such as the variable `x` in the example below, are represented by Theano variables but with extra information attached, such as the associated prior distribution. In PyMC one creates a variable by calling a distribution (its prior distribution) and giving it a name as shown below

.. code:: python
    with Model() as model:
        x = Normal('x', 0,1)

The calls to create a Theano variable (calls to `iscalar`, `dvector`, etc.) are done behind the scenes by PyMC, so you will not use these yourself.  

Once you have a Theano variable, manipulating it is exactly the same in PyMC and Theano, so that part of the tutorial carries over.

The syntax for manipulating variables in PyMC/Theano is quite similar to numpy. Variables can be indexed the same way, and there are array functions like `sum` and `exp` for them. All of Theano's functions are available in PyMC. 

One key distinction between Theano variables and numpy arrays is that Theano variables don't carry out a computation like numpy arrays do, but instead *represent* that computation so it can be manipulated and compiled into an efficient function. With Theano variables, an expression such as `x * y` does not directly carry out a numerical computation. You must first compile the result and then call the resulting function to carry out the computation.

In numpy you would do
.. code:: python
    from numpy import * 
    a = np.array([1,2,3])
    b = 2

    print = a*b
    #prints: [2, 4, 6]

In Theano, you would do
.. code:: python
    from theano.tensor import * 
    a = dvector('a') #this part will be different in PyMC! 
    b = dscalar('b')

    r = a*b
    fn = function([a,b], r)
    print fn([1,2,3], 2)
    #prints: [2, 4, 6]

The `dvector` call creates a Theano variable named 'a' that represents a one dimensional numpy array of doubles. `dscalar` creates one named 'b' that represents a zero dimensional array of doubles.

multiplying the two together creates a Theano variable that represents taking those two numpy arrays and multiplying them together as if they were substituted for a and b. 

The call to `function` creates an actual function that takes two inputs, a one dimensional numpy array and a zero dimensional numpy array and returns the result of the computation represented by `r`. 

In pymc 3, the user does not directly create theano variables. Instead they are returned when you create a random variable (such as by calling `x = Normal('x', 0,1)`). Likewise, you don't call `function` directly usually (though you can), it is used automatically by functions like `model.logp`. 


Differences between Theano and PyMC
-----------------------------------

PyMC variables defined by specifying their prior. Theano variables defined by the functions `dscalar`, `dvector`, `dmatrix` ... .
.. code:: python
    with Model() as model:
        x = Normal('x', 0,1)

Theano variables don't don't have a shape only a number of dimensions and broadcastable but PyMC variables have a shape. 

