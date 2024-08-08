from abc import ABC
from abc import abstractmethod
from fastprogress.fastprogress import master_bar, progress_bar
from functools import partial
import jax.numpy as jnp
import jax.random as jr
from jax import value_and_grad, grad, jit, lax, vmap
from jax.tree_util import tree_map, tree_flatten
from jaxtyping import Float, Array, PyTree
import optax
from tensorflow_probability.substrates.jax import distributions as tfd
from typing import Optional, Union, Tuple, Any
from typing_extensions import Protocol

from dynamax.parameters import to_unconstrained, from_unconstrained
from dynamax.parameters import ParameterSet, PropertySet
from dynamax.types import PRNGKey, Scalar
from dynamax.utils.optimize import run_sgd
from dynamax.utils.utils import ensure_array_has_batch_dim


class Posterior(Protocol):
    """A :class:`NamedTuple` with parameters stored as :class:`jax.DeviceArray` in the leaf nodes."""
    pass

class SuffStatsSSM(Protocol):
    """A :class:`NamedTuple` with sufficient statics stored as :class:`jax.DeviceArray` in the leaf nodes."""
    pass

class SSM(ABC):
    r"""A base class for state space models. Such models consist of parameters, which
    we may learn, as well as hyperparameters, which specify static properties of the
    model. This base class allows parameters to be indicated a standardized way
    so that they can easily be converted to/from unconstrained form for optimization.

    **Abstract Methods**

    Models that inherit from `SSM` must implement a few key functions and properties:

    * :meth:`initial_distribution` returns the distribution over the initial state given parameters
    * :meth:`transition_distribution` returns the conditional distribution over the next state given the current state and parameters
    * :meth:`emission_distribution` returns the conditional distribution over the emission given the current state and parameters
    * :meth:`log_prior` (optional) returns the log prior probability of the parameters
    * :attr:`emission_shape` returns a tuple specification of the emission shape
    * :attr:`inputs_shape` returns a tuple specification of the input shape, or `None` if there are no inputs.

    The shape properties are required for properly handling batches of data.

    **Sampling and Computing Log Probabilities**

    Once these have been implemented, subclasses will inherit the ability to sample
    and compute log joint probabilities from the base class functions:

    * :meth:`sample` draws samples of the states and emissions for given parameters
    * :meth:`log_prob` computes the log joint probability of the states and emissions for given parameters

    **Inference**

    Many subclasses of SSMs expose basic functions for performing state inference.

    * :meth:`marginal_log_prob` computes the marginal log probability of the emissions, summing over latent states
    * :meth:`filter` computes the filtered posteriors
    * :meth:`smoother` computes the smoothed posteriors

    **Learning**

    Likewise, many SSMs will support learning with expectation-maximization (EM) or stochastic gradient descent (SGD).

    For expectation-maximization, subclasses must implement the E- and M-steps.

    * :meth:`e_step` computes the expected sufficient statistics for a sequence of emissions, given parameters
    * :meth:`m_step` finds new parameters that maximize the expected log joint probability

    Once these are implemented, the generic SSM class allows to fit the model with EM

    * :meth:`fit_em` run EM to find parameters that maximize the likelihood (or posterior) probability.

    For SGD, any subclass that implements :meth:`marginal_log_prob` inherits the base class fitting function

    * :meth:`fit_sgd` run SGD to minimize the *negative* marginal log probability.

    """

    @abstractmethod
    def initial_distribution(
        self,
        params: ParameterSet,
        inputs: Optional[Float[Array, "input_dim"]]
    ) -> tfd.Distribution:
        r"""Return an initial distribution over latent states.

        Args:
            params: model parameters $\theta$
            inputs: optional  inputs  $u_t$

        Returns:
            distribution over initial latent state, $p(z_1 \mid \theta)$.

        """
        raise NotImplementedError

    @abstractmethod
    def transition_distribution(
        self,
        params: ParameterSet,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "input_dim"]]
    ) -> tfd.Distribution:
        r"""Return a distribution over next latent state given current state.

        Args:
            params: model parameters $\theta$
            state: current latent state $z_t$
            inputs: current inputs  $u_t$

        Returns:
            conditional distribution of next latent state $p(z_{t+1} \mid z_t, u_t, \theta)$.

        """
        raise NotImplementedError

    @abstractmethod
    def emission_distribution(
        self,
        params: ParameterSet,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "input_dim"]]=None
    ) -> tfd.Distribution:
        r"""Return a distribution over emissions given current state.

        Args:
            params: model parameters $\theta$
            state: current latent state $z_t$
            inputs: current inputs  $u_t$

        Returns:
            conditional distribution of current emission $p(y_t \mid z_t, u_t, \theta)$

        """
        raise NotImplementedError

    def log_prior(
        self,
        params: ParameterSet
    ) -> Scalar:
        r"""Return the log prior probability of any model parameters.

        Returns:
            lp (Scalar): log prior probability.
        """
        return 0.0

    @property
    @abstractmethod
    def emission_shape(self) -> Tuple[int]:
        r"""Return a pytree matching the pytree of tuples specifying the shape of a single time step's emissions.

        For example, a `GaussianHMM` with $D$ dimensional emissions would return `(D,)`.

        """
        raise NotImplementedError

    @property
    def inputs_shape(self) -> Optional[Tuple[int]]:
        r"""Return a pytree matching the pytree of tuples specifying the shape of a single time step's inputs.

        """
        return None

    # All SSMs support sampling
    def sample(
        self,
        params: ParameterSet,
        key: PRNGKey,
        num_timesteps: int,
        inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
    ) -> Tuple[Float[Array, "num_timesteps state_dim"],
              Float[Array, "num_timesteps emission_dim"]]:
        r"""Sample states $z_{1:T}$ and emissions $y_{1:T}$ given parameters $\theta$ and (optionally) inputs $u_{1:T}$.

        Args:
            params: model parameters $\theta$
            key: random number generator
            num_timesteps: number of timesteps $T$
            inputs: inputs $u_{1:T}$

        Returns:
            latent states and emissions

        """
        def _step(prev_state, args):
            key, inpt = args
            key1, key2 = jr.split(key, 2)
            state = self.transition_distribution(params, prev_state, inpt).sample(seed=key2)
            emission = self.emission_distribution(params, state, inpt).sample(seed=key1)
            return state, (state, emission)

        # Sample the initial state
        key1, key2, key = jr.split(key, 3)
        initial_input = tree_map(lambda x: x[0], inputs)
        initial_state = self.initial_distribution(params, initial_input).sample(seed=key1)
        initial_emission = self.emission_distribution(params, initial_state, initial_input).sample(seed=key2)

        # Sample the remaining emissions and states
        next_keys = jr.split(key, num_timesteps - 1)
        next_inputs = tree_map(lambda x: x[1:], inputs)
        _, (next_states, next_emissions) = lax.scan(_step, initial_state, (next_keys, next_inputs))

        # Concatenate the initial state and emission with the following ones
        expand_and_cat = lambda x0, x1T: jnp.concatenate((jnp.expand_dims(x0, 0), x1T))
        states = tree_map(expand_and_cat, initial_state, next_states)
        emissions = tree_map(expand_and_cat, initial_emission, next_emissions)
        return states, emissions

    def log_prob(
        self,
        params: ParameterSet,
        states: Float[Array, "num_timesteps state_dim"],
        emissions: Float[Array, "num_timesteps emission_dim"],
        inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
    ) -> Scalar:
        r"""Compute the log joint probability of the states and observations"""

        def _step(carry, args):
            lp, prev_state = carry
            state, emission, inpt = args
            lp += self.transition_distribution(params, prev_state, inpt).log_prob(state)
            lp += self.emission_distribution(params, state, inpt).log_prob(emission)
            return (lp, state), None

        # Compute log prob of initial time step
        initial_state = tree_map(lambda x: x[0], states)
        initial_emission = tree_map(lambda x: x[0], emissions)
        initial_input = tree_map(lambda x: x[0], inputs)
        lp = self.initial_distribution(params, initial_input).log_prob(initial_state)
        lp += self.emission_distribution(params, initial_state, initial_input).log_prob(initial_emission)

        # Scan over remaining time steps
        next_states = tree_map(lambda x: x[1:], states)
        next_emissions = tree_map(lambda x: x[1:], emissions)
        next_inputs = tree_map(lambda x: x[1:], inputs)
        (lp, _), _ = lax.scan(_step, (lp, initial_state), (next_states, next_emissions, next_inputs))
        return lp

    # Some SSMs will implement these inference functions.
    def marginal_log_prob(
        self,
        params: ParameterSet,
        emissions: Float[Array, "ntime emission_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> Scalar:
        r"""Compute log marginal likelihood of observations, $\log \sum_{z_{1:T}} p(y_{1:T}, z_{1:T} \mid \theta)$.

        Args:
            params: model parameters $\theta$
            state: current latent state $z_t$
            inputs: current inputs  $u_t$

        Returns:
            marginal log probability

        """
        raise NotImplementedError

    def filter(
        self,
        params: ParameterSet,
        emissions: Float[Array, "ntime emission_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> Posterior:
        r"""Compute filtering distributions, $p(z_t \mid y_{1:t}, u_{1:t}, \theta)$ for $t=1,\ldots,T$.

        Args:
            params: model parameters $\theta$
            state: current latent state $z_t$
            inputs: current inputs  $u_t$

        Returns:
            filtering distributions

        """
        raise NotImplementedError

    def smoother(
        self,
        params: ParameterSet,
        emissions: Float[Array, "ntime emission_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> Posterior:
        r"""Compute smoothing distribution, $p(z_t \mid y_{1:T}, u_{1:T}, \theta)$ for $t=1,\ldots,T$.

        Args:
            params: model parameters $\theta$
            state: current latent state $z_t$
            inputs: current inputs  $u_t$

        Returns:
            smoothing distributions

        """
        raise NotImplementedError

    # Learning algorithms
    def e_step(
        self,
        params: ParameterSet,
        emissions: Float[Array, "num_timesteps emission_dim"],
        inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
    ) -> Tuple[SuffStatsSSM, Scalar]:
        r"""Perform an E-step to compute expected sufficient statistics under the posterior, $p(z_{1:T} \mid y_{1:T}, u_{1:T}, \theta)$.

        Args:
            params: model parameters $\theta$
            emissions: emissions $y_{1:T}$
            inputs: optional inputs $u_{1:T}$

        Returns:
            Expected sufficient statistics under the posterior.

        """
        raise NotImplementedError

    def m_step(
        self,
        params: ParameterSet,
        props: PropertySet,
        batch_stats: SuffStatsSSM,
        m_step_state: Any
    ) -> ParameterSet:
        r"""Perform an M-step to find parameters that maximize the expected log joint probability.

        Specifically, compute

        $$\theta^\star = \mathrm{argmax}_\theta \; \mathbb{E}_{p(z_{1:T} \mid y_{1:T}, u_{1:T}, \theta)} \big[\log p(y_{1:T}, z_{1:T}, \theta \mid u_{1:T}) \big]$$

        Args:
            params: model parameters $\theta$
            props: properties specifying which parameters should be learned
            batch_stats: sufficient statistics from each sequence
            m_step_state: any required state for optimizing the model parameters.

        Returns:
            new parameters

        """
        raise NotImplementedError

    def fit_em(
        self,
        params: ParameterSet,
        props: PropertySet,
        emissions: Union[Float[Array, "num_timesteps emission_dim"],
                         Float[Array, "num_batches num_timesteps emission_dim"]],
        inputs: Optional[Union[Float[Array, "num_timesteps input_dim"],
                               Float[Array, "num_batches num_timesteps input_dim"]]]=None,
        num_iters: int=50,
        verbose: bool=True
    ) -> Tuple[ParameterSet, Float[Array, "num_iters"]]:
        r"""Compute parameter MLE/ MAP estimate using Expectation-Maximization (EM).

        EM aims to find parameters that maximize the marginal log probability,

        $$\theta^\star = \mathrm{argmax}_\theta \; \log p(y_{1:T}, \theta \mid u_{1:T})$$

        It does so by iteratively forming a lower bound (the "E-step") and then maximizing it (the "M-step").

        *Note:* ``emissions`` *and* ``inputs`` *can either be single sequences or batches of sequences.*

        Args:
            params: model parameters $\theta$
            props: properties specifying which parameters should be learned
            emissions: one or more sequences of emissions
            inputs: one or more sequences of corresponding inputs
            num_iters: number of iterations of EM to run
            verbose: whether or not to show a progress bar

        Returns:
            tuple of new parameters and log likelihoods over the course of EM iterations.

        """

        # Make sure the emissions and inputs have batch dimensions
        batch_emissions = ensure_array_has_batch_dim(emissions, self.emission_shape)
        batch_inputs = ensure_array_has_batch_dim(inputs, self.inputs_shape)

        @jit
        def em_step(params, m_step_state):
            batch_stats, lls = vmap(partial(self.e_step, params))(batch_emissions, batch_inputs)
            lp = self.log_prior(params) + lls.sum()
            params, m_step_state = self.m_step(params, props, batch_stats, m_step_state)
            # debug.print('e_step: {x}', x=(batch_stats, lls))
            # debug.print('m_step{y}', y=params)
            return params, m_step_state, lp

        log_probs = []
        m_step_state = self.initialize_m_step_state(params, props)
        pbar = progress_bar(range(num_iters)) if verbose else range(num_iters)
        for _ in pbar:
            params, m_step_state, marginal_loglik = em_step(params, m_step_state)
            log_probs.append(marginal_loglik)
        return params, jnp.array(log_probs)


    def fit_em_stoch_sessions(
            self,
            params: ParameterSet,
            props: PropertySet,
            all_session_emissions: [],
            all_session_inputs: [],
            num_iters: int=50,
            forgetting_rate: float=-0.5,
            key: PRNGKey=jr.PRNGKey(0),
            verbose: bool=True
    ) -> Tuple[ParameterSet, Float[Array, "num_iters"]]:
            r"""Stochastic Expectation Maximization (EM) with mini-batches and dynamic step size.
            Args:
                params: model parameters $\theta$
                props: properties specifying which parameters should be learned
                emissions: one or more sequences of emissions
                inputs: one or more sequences of corresponding inputs
                mini_batch_size: number of sequences per mini-batch
                num_iters: number of iterations of stochastic EM to run
                forgetting_rate: rate to adjust the learning step size
                key: random number generator for selecting minibatches
                verbose: whether or not to show a progress bar
            Returns:
                tuple of new parameters and log likelihoods over the course of EM iterations.
            """
            num_sessions=int(len(all_session_emissions))

            # ensure sessions have batch form
            all_session_params=[]
            for ses in range(num_sessions):
                if jnp.ndim(all_session_emissions[ses])==2:
                    all_session_emissions[ses] = jnp.expand_dims(all_session_emissions[ses], axis=0)
                    all_session_inputs[ses] = jnp.expand_dims(all_session_inputs[ses], axis=0) if all_session_inputs is not None else None
                # hierarchical EM: optimize mix between session/subject and group effects: all_session_params.append(params)
                all_session_params.append(params)
                
            # Ensure emissions and inputs have batch dimensions
            all_session_sizes=[ses.shape[0] for ses in all_session_emissions]
            dataset_size=int(jnp.sum(jnp.array(all_session_sizes)))
            max_session_size=int(jnp.max(jnp.array(all_session_sizes)))

            # Initialize the step size schedule
            schedule = jnp.arange(2, 1 + (1 + num_sessions) * num_iters) ** forgetting_rate
                   
            @jit
            def update_global_struct_session(global_struct, current_struct, step_size, dataset_size, current_session_size):
                rescale = lambda x: (dataset_size / current_session_size) * x
                rescaled_struct = tree_map(rescale, current_struct)
                blend = lambda g, b: g * (1 - step_size) + b * step_size
                return tree_map(blend, global_struct, rescaled_struct)

            @jit
            def em_stoch_step_session(emissions, inputs, params, m_step_state, global_stats, subiter_idx, current_session_size):

                # E-step: Compute expected sufficient statistics for this mini-batch
                session_stats, lls = vmap(partial(self.e_step, params))(emissions, inputs)
                                
                # Blend batch statistics into global statistics using the same optimizer as twarhmm
                step_size = schedule[subiter_idx]
                
                # flatten stats
                flat_session_stats = tree_map(lambda x: jnp.sum(x, axis=0), session_stats)
                                
                # update global stats (with rescaling)
                global_stats = update_global_struct_session(global_stats, flat_session_stats, step_size, dataset_size, current_session_size)
                
                # create a batch representation, duplicating as needed
                #batched_global_stats = tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, axis=0),current_session_size,axis=0, total_repeat_length=max_session_size), global_stats)
                
                batched_global_stats = tree_map(lambda x: jnp.expand_dims(x, axis=0), global_stats)
                
                # Log probability update before m step
                lp = self.log_prior(params)*(current_session_size/dataset_size) + lls.sum() # unsure whether lls.sum shoud be scaled

                # M-step: Update parameters using blended global statistics
                params, m_step_state = self.m_step(params, props, batched_global_stats, m_step_state)
               
               # self.m_step(params, props, batched_global_stats, m_step_state)
                                       
                return params, m_step_state, lp, global_stats, session_stats

            # Initialize log probability list and M-step state
            log_probs = []

            if verbose:
                pbar = progress_bar(range(num_iters)) if verbose else range(num_iters)
            else:
                pbar = range(num_iters)

            pbar = progress_bar(range(num_iters)) if verbose else range(num_iters)

            m_step_state = self.initialize_m_step_state(params, props)
            
            init_params = params

            for iter_idx in pbar:

                # Shuffle the batches at each iteration
                key_perm, key = jr.split(key, 2)
                shuffled_indices = jr.permutation(key_perm, num_sessions)
                session_emissions=[all_session_emissions[ind] for ind in shuffled_indices]
                session_sizes=[all_session_sizes[ind] for ind in shuffled_indices]
                session_inputs=[all_session_inputs[ind] for ind in shuffled_indices] if all_session_sizes is not None else None
                #session_params=[all_session_params[ind] for ind in shuffled_indices]

                session_counter=0
                marginal_loglik=0

                for emissions, inputs, current_session_size in zip(session_emissions,session_inputs, session_sizes):
                                        
                    subiter_idx = iter_idx * num_sessions + session_counter

                    if iter_idx==0:
                        init_stats, lls = vmap(partial(self.e_step, init_params))(emissions, inputs)
                        global_stats = tree_map(lambda x: jnp.sum(x, axis=0), init_stats)
                    else:
                        reinject_prior=1.0-(10**-7)
                        params=self.update_params(prior_params=init_params, params=params, up_initial=reinject_prior, up_emissions_covs=reinject_prior, up_emissions_weights=reinject_prior)
                        #params = update_global_struct_session(params, init_params, schedule[-1]/10000.0, 1, 1)

                    #keep_params=params
                    #keep_stats=global_stats
                    #keep_m_state=m_step_state
                    #keep_marginal_loglik_session=lls
                    params, m_step_state, marginal_loglik_session, global_stats, flat_session_stats = em_stoch_step_session(emissions, inputs, params, m_step_state, global_stats, subiter_idx, current_session_size)

                    #if jnp.isnan(marginal_loglik_session):
                    #    return init_stats, keep_params,keep_stats,params,global_stats, subiter_idx, emissions, inputs, flat_session_stats, current_session_size, keep_m_state, m_step_state, dataset_size, keep_marginal_loglik_session
                    
                    marginal_loglik += marginal_loglik_session

                    session_counter += 1

                log_probs.append(marginal_loglik)


            return params, jnp.array(log_probs)
        

    def fit_em_cv(
        self,
        params: ParameterSet,
        props: PropertySet,
        emissions: Union[Float[Array, "num_timesteps emission_dim"], Float[Array, "num_batches num_timesteps emission_dim"]],
        inputs: Optional[Union[Float[Array, "num_timesteps input_dim"], Float[Array, "num_batches num_timesteps input_dim"]]] = None,
        num_iters: int = 50,
        num_folds: int = 3,
        verbose: bool = True
    ) -> Tuple[ParameterSet, Float[Array, "num_iters"]]:
        r"""Compute parameter MLE/ MAP estimate using Expectation-Maximization (EM) with cross-validation.

        EM aims to find parameters that maximize the marginal log probability,

        $$\theta^\star = \mathrm{argmax}_\theta \; \log p(y_{1:T}, \theta \mid u_{1:T})$$

        It does so by iteratively forming a lower bound (the "E-step") and then maximizing it (the "M-step").

        *Note:* ``emissions`` and ``inputs`` must already be formatted as batches of sequences, since the cross-validation will split these
        batches as training and testing sets.

        Args:
            params: Model parameters $\theta$.
            props: Properties specifying which parameters should be learned.
            emissions: One or more sequences of emissions.
            inputs: One or more sequences of corresponding inputs (optional).
            num_iters: Number of iterations of EM to run (default: 50).
            num_folds: Number of folds for cross-validation (default: 3).
            verbose: Whether or not to show a progress bar (default: True).

        Returns:
            tuple: A tuple of parameters (averaged parameters over folds) and log likelihoods over the course of EM iterations.
        """
        from sklearn.model_selection import KFold
        
        def tree_unstack(tree):
            leaves, treedef = tree_flatten(tree)
            n_trees = leaves[0].shape[0]
            new_leaves = [[] for _ in range(n_trees)]
            for leaf in leaves:
                for i in range(n_trees):
                    new_leaves[i].append(leaf[i])
            new_trees = [treedef.unflatten(l) for l in new_leaves]
            return new_trees

        if jnp.ndim(emissions) != 3 or (inputs is not None and jnp.ndim(inputs) != 3):
            raise ValueError('The data should be preformatted in nbatch_timestep_emissiondim form.')

        kf = KFold(n_splits=num_folds)
        train_size = int(emissions.shape[0]-(jnp.ceil(emissions.shape[0]/num_folds)))
        fold_batch_emissions = jnp.zeros((num_folds, train_size, emissions.shape[1], emissions.shape[2]))
        fold_batch_inputs = jnp.zeros((num_folds, train_size, inputs.shape[1], inputs.shape[2])) if inputs is not None else None
        fold_batch_emissions_test = []
        fold_batch_inputs_test = []
        
        for f, (train_ind, test_ind) in enumerate(kf.split(emissions)):
            fold_batch_emissions=fold_batch_emissions.at[f, :, :, :].set(emissions[train_ind[:train_size], :, :])
            fold_batch_emissions_test.append(emissions[test_ind, :, :])
            if inputs is not None:
                fold_batch_inputs=fold_batch_inputs.at[f, :, :, :].set(inputs[train_ind[:train_size], :, :])
                fold_batch_inputs_test.append(inputs[test_ind, :, :])
        
        fold_params = [params for _ in range(num_folds)]
        fold_params = tree_map(lambda *x: jnp.stack(x), *fold_params)

        @jit
        def em_step_data(params, m_step_state, emissions, inputs):
            batch_stats, lls = vmap(partial(self.e_step, params))(emissions, inputs)
            lp = self.log_prior(params) + lls.sum()
            params, m_step_state = self.m_step(params, props, batch_stats, m_step_state)
            return params, m_step_state, lp

        @jit
        def em_step_folds(fold_params, fold_batch_emissions, fold_batch_inputs, m_step_state):
            fold_params, m_step_state, lp = vmap(
                lambda params, emissions, inputs: em_step_data(params, m_step_state, emissions, inputs)
            )(fold_params, fold_batch_emissions, fold_batch_inputs)
            return fold_params, m_step_state, lp
        
        log_probs = []
        log_probs_test = []
        m_step_state = self.initialize_m_step_state(params, props)
        pbar = progress_bar(range(num_iters)) if verbose else range(num_iters)
        
        for _ in pbar:
            fold_params, m_step_state, marginal_loglik = em_step_folds(fold_params, fold_batch_emissions, fold_batch_inputs, m_step_state)
            unstacked_fold_params = tree_unstack(fold_params)
            test_lls = 0.0
            for f in range(num_folds):
                _, lls = vmap(partial(self.e_step, unstacked_fold_params[f]))(fold_batch_emissions_test[f], fold_batch_inputs_test[f])
                test_lls += lls.sum()
            log_probs.append(marginal_loglik)
            log_probs_test.append(test_lls)
        
        params = tree_map(lambda x: jnp.mean(x, axis=0), fold_params)
        return params, log_probs, jnp.array(log_probs_test)
    
    def fit_em_cv_prior_constrain(
        self,
        params: ParameterSet,
        props: PropertySet,
        emissions: Union[Float[Array, "num_timesteps emission_dim"], Float[Array, "num_batches num_timesteps emission_dim"]],
        inputs: Optional[Union[Float[Array, "num_timesteps input_dim"], Float[Array, "num_batches num_timesteps input_dim"]]] = None,
        num_iters_cv: int = 50,
        num_iters_mixing: int = 50,
        num_iters_final: int = 50,
        num_folds: int = 3,
        constrain_initial: float=1.0,
        constrain_transitions: float=1.0,
        constrain_emissions: float=1.0,
        forgetting_rate: float=-0.5,
        global_stoch_em: bool=False,
        verbose: bool = True
    ) -> Tuple[ParameterSet, Float[Array, "num_iters"]]:
        r"""Compute parameter MLE/ MAP estimate using Expectation-Maximization (EM) with cross-validation.

        EM aims to find parameters that maximize the marginal log probability,

        $$\theta^\star = \mathrm{argmax}_\theta \; \log p(y_{1:T}, \theta \mid u_{1:T})$$

        It does so by iteratively forming a lower bound (the "E-step") and then maximizing it (the "M-step").

        *Note:* ``emissions`` and ``inputs`` must already be formatted as batches of sequences, since the cross-validation will split these
        batches as training and testing sets.

        Args:
            params: Model parameters $\theta$.
            props: Properties specifying which parameters should be learned.
            emissions: One or more sequences of emissions.
            inputs: One or more sequences of corresponding inputs (optional).
            num_iters: Number of iterations of EM to run (default: 50).
            num_folds: Number of folds for cross-validation (default: 3).
            verbose: Whether or not to show a progress bar (default: True).

        Returns:
            tuple: A tuple of parameters (averaged parameters over folds) and log likelihoods over the course of EM iterations.
        """
        from sklearn.model_selection import KFold
        from jax import nn as jnn
        import numpy as np
                
        @jit
        def em_step_data(params, m_step_state, emissions, inputs):
            batch_stats, lls = vmap(partial(self.e_step, params))(emissions, inputs)
            lp = self.log_prior(params) + lls.sum()
            params, m_step_state = self.m_step(params, props, batch_stats, m_step_state)
            return params, m_step_state, lp

        @jit
        def em_step_folds(fold_params, fold_batch_emissions, fold_batch_inputs, m_step_state):
            fold_params, m_step_state, lp = vmap(
                lambda params, emissions, inputs: em_step_data(params, m_step_state, emissions, inputs)
            )(fold_params, fold_batch_emissions, fold_batch_inputs)
            return fold_params, m_step_state, lp
          

        prior_params=params
        
        def tree_unstack(tree):
            leaves, treedef = tree_flatten(tree)
            n_trees = leaves[0].shape[0]
            new_leaves = [[] for _ in range(n_trees)]
            for leaf in leaves:
                for i in range(n_trees):
                    new_leaves[i].append(leaf[i])
            new_trees = [treedef.unflatten(l) for l in new_leaves]
            return new_trees

        if jnp.ndim(emissions) != 3 or (inputs is not None and jnp.ndim(inputs) != 3):
            raise ValueError('The data should be preformatted in nbatch_timestep_emissiondim form.')

        kf = KFold(n_splits=num_folds)
        train_size = int(emissions.shape[0]-(jnp.ceil(emissions.shape[0]/num_folds)))
        fold_batch_emissions = jnp.zeros((num_folds, train_size, emissions.shape[1], emissions.shape[2]))
        fold_batch_inputs = jnp.zeros((num_folds, train_size, inputs.shape[1], inputs.shape[2])) if inputs is not None else None
        fold_batch_emissions_test = []
        fold_batch_inputs_test = []
        
        for f, (train_ind, test_ind) in enumerate(kf.split(emissions)):
            fold_batch_emissions=fold_batch_emissions.at[f, :, :, :].set(emissions[train_ind[:train_size], :, :])
            fold_batch_emissions_test.append(emissions[test_ind, :, :])
            if inputs is not None:
                fold_batch_inputs=fold_batch_inputs.at[f, :, :, :].set(inputs[train_ind[:train_size], :, :])
                fold_batch_inputs_test.append(inputs[test_ind, :, :])
        
        fold_params = [params for _ in range(num_folds)]
        fold_params = tree_map(lambda *x: jnp.stack(x), *fold_params)
                
        
        log_probs_train = []
        log_probs_test = []
        m_step_state = self.initialize_m_step_state(params, props)
        pbar_cv = master_bar(range(num_iters_cv)) if verbose else range(num_iters_cv)
        pbar_cv.names = ['testLL', 'trainLL']
        x_bounds = [0, num_iters_cv]
        for iter_idx in pbar_cv:
            fold_params, m_step_state, marginal_loglik = em_step_folds(fold_params, fold_batch_emissions, fold_batch_inputs, m_step_state)
            unstacked_fold_params = tree_unstack(fold_params)
            unmixed_lls = 0.0
            unmixed_count=0
            for f in range(num_folds):
                _, fold_lls = vmap(partial(self.e_step, unstacked_fold_params[f]))(fold_batch_emissions_test[f], fold_batch_inputs_test[f])
                unmixed_lls+=fold_lls.sum()
                unmixed_count+=len(fold_lls)
            log_probs_train.append(marginal_loglik.sum())
            log_probs_test.append(unmixed_lls)
            if iter_idx>1 and verbose:
                graphs = [[np.cumsum(np.ones((iter_idx+1,))),np.array(log_probs_test)/(unmixed_count)],[np.cumsum(np.ones((iter_idx+1,))),np.array(log_probs_train)/(train_size*num_folds)]]
                y_bounds = [np.percentile((np.array(graphs)[:,1]),[10])[0],np.max(np.array(graphs)[:,1])*1.1]
                pbar_cv.update_graph(graphs, x_bounds,y_bounds)
            
                    
        def loss_fn(prior_params, params, m_step_state, emissions, inputs, upweight_initial, upweight_transitions, upweight_emissions):
            mixed_param=self.update_params(prior_params=prior_params, params=params, up_initial=constrain_initial*jnn.sigmoid(upweight_initial), up_transitions=constrain_transitions*jnn.sigmoid(upweight_transitions), up_emissions_biases=constrain_emissions*jnn.sigmoid(upweight_emissions), up_emissions_covs=constrain_emissions*jnn.sigmoid(upweight_emissions), up_emissions_weights=constrain_emissions*jnn.sigmoid(upweight_emissions))      
            _, inner_lls = vmap(partial(self.e_step, mixed_param))(emissions, inputs)
            return -inner_lls.sum()
        
        schedule_mix = jnp.arange(2, 1 + num_iters_mixing) ** (forgetting_rate)
        convergence_threshold=10**-2
        grad_threshold=10**-3
        upweight_initial, upweight_transitions, upweight_emissions = 0.0, 0.0, 0.0
        
        #fig, ax = plt.subplots( figsize=(4, 4))
        #ax.plot(schedule_mix)

        mb = master_bar(range(num_folds))
        mb.names = [f'fold {f}' for f in range(num_folds)]
        x_bounds = [0, num_iters_mixing]
        pbar = progress_bar(range(num_iters_mixing), parent=mb) if verbose else range(num_iters_mixing)
        upweight_initial_fold=[upweight_initial]
        upweight_transitions_fold=[upweight_transitions]
        upweight_emissions_fold=[upweight_emissions]
        mixed_lls=np.zeros((num_folds, num_iters_mixing))*np.nan
        for f in mb:
            if f>0:
                upweight_initial=np.mean(np.array(upweight_initial_fold)[1:len(upweight_initial_fold)+1])
            upweight_transitions=upweight_transitions_fold[0]
            upweight_emissions=upweight_emissions_fold[0]
            prev_mixed_lls = float('inf')
            grad_transitions, grad_emissions, grad_initial = 0.0, 0.0, 0.0
            for mixing_iter_idx in pbar:
                fold_mixed_lls, mix_grad = value_and_grad(loss_fn, argnums=(5, 6, 7))(prior_params, unstacked_fold_params[f], m_step_state, fold_batch_emissions_test[f], fold_batch_inputs_test[f], upweight_initial, upweight_transitions, upweight_emissions)
                mixed_lls[f,mixing_iter_idx]=fold_mixed_lls.sum()
                grad_initial=+mix_grad[0]
                grad_transitions =+ mix_grad[1]
                grad_emissions =+ mix_grad[2]
                upweight_initial=upweight_initial - schedule_mix[mixing_iter_idx] * grad_initial
                upweight_transitions=upweight_transitions - schedule_mix[mixing_iter_idx] * grad_transitions
                upweight_emissions=upweight_emissions - schedule_mix[mixing_iter_idx] * (grad_emissions/num_folds)
                if jnp.abs(prev_mixed_lls - fold_mixed_lls.sum())<convergence_threshold:
                    mb.main_bar.comment=f"Fold {f} converged"
                    break
                else:
                    prev_mixed_lls=fold_mixed_lls.sum()
                if mixing_iter_idx>1 and verbose:
                    graphs = [[np.cumsum(np.ones((num_iters_mixing,))),mixed_lls[ff,:]] for ff in range(num_folds)]
                    y_bounds=[-np.abs(np.nanmin(mixed_lls))*1.1,np.nanmax(mixed_lls)*1.1]
                    bar_msg=f"up_init={constrain_initial*jnn.sigmoid(upweight_initial): .2f} / up_trans={constrain_transitions*jnn.sigmoid(upweight_transitions): .2f} / up_em={constrain_emissions*jnn.sigmoid(upweight_emissions): .2f}"
                    mb.update_graph(graphs, x_bounds, y_bounds)
                    mb.child.comment=bar_msg
            if jnp.abs(prev_mixed_lls - fold_mixed_lls.sum())>=convergence_threshold:
                mb.main_bar.comment=f"Fold {f} did not converge"
        
            upweight_initial_fold.append(upweight_initial)
            upweight_transitions_fold.append(upweight_transitions)
            upweight_emissions_fold.append(upweight_emissions)
            
        upweight_initial=jnp.mean(jnp.array(upweight_initial_fold)[1:num_folds])
        upweight_transitions=jnp.mean(jnp.array(upweight_transitions_fold)[1:num_folds])
        upweight_emissions=jnp.mean(jnp.array(upweight_emissions_fold)[1:num_folds])

        
        # averages the folds
        mean_fold_params = tree_map(lambda x: jnp.mean(x, axis=0), fold_params)
        # ensure normalization
        mean_fold_params=self.update_params(prior_params=mean_fold_params, params=mean_fold_params)      
        
        # fit the model on the whole data
        if global_stoch_em:
            free_params, lls_free=self.fit_em_stoch_sessions(mean_fold_params, props, emissions=emissions,inputs=inputs, num_iters=num_iters_final)
        else:
            free_params, lls_free=self.fit_em(mean_fold_params, props, emissions=emissions,inputs=inputs, num_iters=num_iters_final)
        
        # mix the params according to the best mixing parameters found above
        mixed_params=self.update_params(prior_params=prior_params, params=free_params, up_initial=constrain_initial*jnn.sigmoid(upweight_initial), up_transitions=constrain_transitions*jnn.sigmoid(upweight_transitions), up_emissions_biases=constrain_emissions*jnn.sigmoid(upweight_emissions), up_emissions_covs=constrain_emissions*jnn.sigmoid(upweight_emissions),up_emissions_weights=constrain_emissions*jnn.sigmoid(upweight_emissions))
        
        return free_params, lls_free, (mixed_params, jnp.array(log_probs_train), jnp.array(log_probs_test), mixed_lls, upweight_initial_fold, mix_grad)

    def fit_sgd(
        self,
        params: ParameterSet,
        props: PropertySet,
        emissions: Union[Float[Array, "num_timesteps emission_dim"],
                         Float[Array, "num_batches num_timesteps emission_dim"]],
        inputs: Optional[Union[Float[Array, "num_timesteps input_dim"],
                               Float[Array, "num_batches num_timesteps input_dim"]]]=None,
        optimizer: optax.GradientTransformation=optax.adam(1e-3),
        batch_size: int=1,
        num_epochs: int=50,
        shuffle: bool=False,
        key: PRNGKey=jr.PRNGKey(0)
    ) -> Tuple[ParameterSet, Float[Array, "niter"]]:
        r"""Compute parameter MLE/ MAP estimate using Stochastic Gradient Descent (SGD).

        SGD aims to find parameters that maximize the marginal log probability,

        $$\theta^\star = \mathrm{argmax}_\theta \; \log p(y_{1:T}, \theta \mid u_{1:T})$$

        by minimizing the _negative_ of that quantity.

        *Note:* ``emissions`` *and* ``inputs`` *can either be single sequences or batches of sequences.*

        On each iteration, the algorithm grabs a *minibatch* of sequences and takes a gradient step.
        One pass through the entire set of sequences is called an *epoch*.

        Args:
            params: model parameters $\theta$
            props: properties specifying which parameters should be learned
            emissions: one or more sequences of emissions
            inputs: one or more sequences of corresponding inputs
            optimizer: an `optax` optimizer for minimization
            batch_size: number of sequences per minibatch
            num_epochs: number of epochs of SGD to run
            key: a random number generator for selecting minibatches
            verbose: whether or not to show a progress bar

        Returns:
            tuple of new parameters and losses (negative scaled marginal log probs) over the course of SGD iterations.

        """
        # Make sure the emissions and inputs have batch dimensions
        batch_emissions = ensure_array_has_batch_dim(emissions, self.emission_shape)
        batch_inputs = ensure_array_has_batch_dim(inputs, self.inputs_shape)

        unc_params = to_unconstrained(params, props)

        def _loss_fn(unc_params, minibatch):
            """Default objective function."""
            params = from_unconstrained(unc_params, props)
            minibatch_emissions, minibatch_inputs = minibatch
            scale = len(batch_emissions) / len(minibatch_emissions)
            minibatch_lls = vmap(partial(self.marginal_log_prob, params))(minibatch_emissions, minibatch_inputs)
            lp = self.log_prior(params) + minibatch_lls.sum() * scale
            return -lp / batch_emissions.size

        dataset = (batch_emissions, batch_inputs)
        unc_params, losses = run_sgd(_loss_fn,
                                     unc_params,
                                     dataset,
                                     optimizer=optimizer,
                                     batch_size=batch_size,
                                     num_epochs=num_epochs,
                                     shuffle=shuffle,
                                     key=key)

        params = from_unconstrained(unc_params, props)
        return params, losses
