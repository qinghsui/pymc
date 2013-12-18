


Pymc 3 is built around Theano, a computational graph package.  

Random random variables, such as the variable `x` in the example below, are represented by Theano variables with extra information attached, such as the associated prior distribution. The syntax for manipulating Theano variables is quite similar to numpy. Theano variables can be indexed the same way, and there are array functions like `sum` and `exp` for them. All of Theano's functions are available in pymc. 

    with Model() as model:
        x = Normal('x', 0,1)

One key distinction between Theano variables and numpy arrays is that with Theano variables expressions like `x * y` do not directly carry out a numerical computation. Instead the result describes the computation, so that it can be compiled into an efficient function for doing the calculation or otherwise manipulated. 

    from theano.tensor import * 
    a = dvector('a')
    b = dscalar('b')

    r = a*b
    fn = function([a,b], r)
    print fn([1,2,3], 2)
    #prints: [2, 4, 6]

The `dvector` call creates a Theano variable named 'a' that represents a one dimensional numpy array of doubles. `dscalar` creates one named 'b' that represents a zero dimensional array of doubles.

multiplying the two together creates a Theano variable that represents taking those two numpy arrays and multiplying them together as if they were substituted for a and b. 

The call to `function` creates an actual function that takes two inputs, a one dimensional numpy array and a zero dimensional numpy array and returns the result of the computation represented by `r`. 

In pymc 3, the user does not directly create theano variables. Instead they are returned when you create a random variable (such as by calling `x = Normal('x', 0,1)`). Likewise, you don't call `function` directly usually (though you can), you let it be automatically called when you use `model.logp`. 
