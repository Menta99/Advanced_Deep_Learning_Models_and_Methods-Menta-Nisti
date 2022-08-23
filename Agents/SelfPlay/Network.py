from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout, Flatten, Input, Conv2D, LeakyReLU, add
from keras.models import Sequential, load_model, Model
import keras.backend as K
from tensorflow.keras.optimizers import Adam


class SelfPlayNetwork:
    def __init__(self, env, learning_rate):
        self.name = env.name
        if env.name == "Santorini":
            self.observation_space = (5, 5, 6) if env.representation == "Tabular" else (160, 160, 1)
            self.action_space = 128
        elif env.name == "TicTacToe":
            self.observation_space = (3, 3, 1) if env.representation == "Tabular" else (96, 96, 1)
            self.action_space = 9
        elif env.name == "ConnectFour":
            self.observation_space = (6, 7, 1) if env.representation == "Tabular" else (192, 224, 1)
            self.action_space = 7
        self.hidden_layers = [
            {'filters': 256, 'kernel_size': (3, 3)},
            {'filters': 256, 'kernel_size': (3, 3)},
            {'filters': 256, 'kernel_size': (3, 3)},
            {'filters': 256, 'kernel_size': (3, 3)},
            {'filters': 256, 'kernel_size': (3, 3)},
            {'filters': 256, 'kernel_size': (3, 3)},
            {'filters': 256, 'kernel_size': (3, 3)},
            {'filters': 256, 'kernel_size': (3, 3)},
            {'filters': 256, 'kernel_size': (3, 3)},
            {'filters': 256, 'kernel_size': (3, 3)}
        ]
        self.num_layers = len(self.hidden_layers)
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        input = Input(shape=self.observation_space, name='input')

        x = self.conv_layer(input, self.hidden_layers[0]['filters'], self.hidden_layers[0]['kernel_size'])

        if len(self.hidden_layers) > 1:
            for h in self.hidden_layers[1:]:
                x = self.residual_layer(x, h['filters'], h['kernel_size'])

        vh = self.value_head(x)
        ph = self.policy_head(x)

        model = Model(inputs=[input], outputs=[vh, ph])
        model.compile(loss={'policy_head' : 'categorical_crossentropy', 'value_head' : 'mean_squared_error'},
                      optimizer=Adam(learning_rate=self.learning_rate),
                      loss_weights={'policy_head': 0.5, 'value_head': 0.5}
                      )

        return model

    def residual_layer(self, input_block, filters, kernel_size):

        x = self.conv_layer(input_block, filters, kernel_size)

        x = Conv2D(
            filters=filters
            , kernel_size=kernel_size
            , padding='same'
            , activation='linear'
        )(x)

        x = BatchNormalization(axis=1)(x)

        x = add([input_block, x])

        x = LeakyReLU()(x)

        return x

    def conv_layer(self, x, filters, kernel_size):

        x = Conv2D(
            filters=filters
            , kernel_size=kernel_size
            , padding='same'
            , activation='linear'
        )(x)

        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        return x

    def value_head(self, x):

        x = Conv2D(
            filters=1
            , kernel_size=(1, 1)
            , padding='same'
            , activation='linear'
        )(x)

        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        x = Flatten()(x)

        x = Dense(20, activation='linear')(x)

        x = LeakyReLU()(x)

        x = Dense(1, activation='tanh', name='value_head')(x)

        return x

    def policy_head(self, x):

        x = Conv2D(
            filters=2
            , kernel_size=(1, 1)
            , padding='same'
            , activation='linear'
        )(x)

        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        x = Flatten()(x)

        x = Dense(
            self.action_space
            , activation='linear'
            , name='policy_head'
        )(x)

        return x

    def predict(self, x):
        return self.model.predict(x=x, verbose=0)

    def fit(self, states, targets, epochs, verbose, validation_split, batch_size):
        return self.model.fit(states, targets, epochs=epochs, verbose=verbose, validation_split=validation_split,
                              batch_size=batch_size)
