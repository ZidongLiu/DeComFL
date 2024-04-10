from optimizers.perturbation_direction_descent import PDD


class PDD2Models:
    def __init__(self, model1, model2, lr, mu, criterion, grad_estimate_method):
        self.model1 = model1
        self.model2 = model2
        self.lr = lr
        self.mu = mu
        self.criterion = criterion

        self.pdd1 = PDD(
            model1.parameters(), lr=lr, mu=mu, grad_estimate_method=grad_estimate_method
        )
        self.pdd2 = PDD(
            model2.parameters(), lr=lr, mu=mu, grad_estimate_method=grad_estimate_method
        )

    def step(self, batch_input, batch_labels):
        self.pdd1.generate_perturbation()
        self.pdd1.apply_perturbation_1()
        perturbation_1_output_from_model1 = self.model1(batch_input)
        self.pdd1.apply_perturbation_2()
        perturbation_2_output_from_model1 = self.model1(batch_input)
        # print(
        #     (perturbation_2_output_from_model1 - perturbation_1_output_from_model1)
        #     .abs()
        #     .sum()
        # )
        # model 2 and calulate loss and grad
        self.pdd2.generate_perturbation()
        self.pdd2.apply_perturbation_1()
        perturbation_1_pred = self.model2(perturbation_1_output_from_model1)
        self.pdd2.apply_perturbation_2()
        perturbation_2_pred = self.model2(perturbation_2_output_from_model1)

        perturbation_1_loss = self.criterion(perturbation_1_pred, batch_labels)
        perturbation_2_loss = self.criterion(perturbation_2_pred, batch_labels)
        # print(perturbation_1_loss, perturbation_2_loss)
        grad = self.pdd2.calculate_grad(perturbation_1_loss, perturbation_2_loss)

        # update model
        self.pdd2.step(grad)
        self.pdd1.step(grad)

        return grad
