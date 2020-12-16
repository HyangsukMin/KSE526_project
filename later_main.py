
class Config():
    def __init__(
        self,
        input_width = 7,
        label_width = 7,
        shift = 7,
        label_columns = ["Maximum_Power_This_Year"],
        batch_size = 32,
        featrues = ["meteo", "covid", "gas", "exchange"],
        filters = 64,
        kernel_size = 3, 
        activation = 'relu',
        lstm_units = 30,
        learning_rate = 0.001,
        epochs = 100,
        ):
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.label_columns = label_columns
        self.batch_size = batch_size
        self.features = features
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        self.epochs = epochs
config = Config()
#%%
@tf.function
def print_status_bar(iteration, total, loss, metrics = None):
    metrics = " - ".join(["{}: {:4.f}".format(m.name, m.result())
                        for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"
    print("\r{}/{} - ".format(iteration, total) + metrics, end=end)

@tf.function
def train(config, model, optimizer, loss_fn, metrics, mean_loss, n_epochs):
    Dataset = WindowGenerator(
                            input_width = config.input_width, 
                            label_width = config.label_width, 
                            shift = config.shift, 
                            label_columns = config.label_columns,
                            batch_size = config.batch_size,
                            features = config.features
                            )
    # w = tf.Variable(initial_value = 10.0, trainable=True, name="loss_weight", dtype=tf.float64)
    for epoch in range(1, n_epochs +1):
        print("Epoch {}/{}".format(epoch, n_epochs))
        train_set = Dataset.train
        for X_batch ,y_batch in train_set:
            with tf.GradientTape(persistent=True) as tape:
                y_pred = model(X_batch)
                main_loss = loss_fn(y_batch, y_pred)#, w)
                loss = tf.add_n([main_loss] + model.losses)
            grads_main = tape.gradient(loss, model.trainable_variables)
            # grads_sub = tape.gradient(loss, w)
            optimizer.apply_gradients(zip(grads_main, model.trainable_variables))
            # optimizer.apply_gradients(zip(grads_sub, w))
            # del tape
            mean_loss(loss)
            for metric in metrics:
                metric(y_batch, y_pred)
            print_status_bar(epoch, n_epochs, mean_loss, metrics)
        print_status_bar(epoch, n_epochs, mean_loss, metrics)
        for metric in [mean_loss] + metrics:
            metric.reset_states()
#%%
optimizer = tf.keras.optimizers.Adam(learning_rate = config.learning_rate)
loss_fn = tf.keras.losses.mean_squared_error #root_mean_squared_error
mean_loss = tf.keras.metrics.Mean()
metrics = [root_mean_squared_error, last_time_step_rmse]
model_mcge = CNNLSTMATTN(config)
#%%
train(config, model_mcge, optimizer, loss_fn, mean_loss, metrics, config.epochs)