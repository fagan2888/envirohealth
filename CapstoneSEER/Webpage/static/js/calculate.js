$(function() {
    $('#btncalculate').click(function() {

        $.ajax({
            url: '/showResults',
            data: $('form').serialize(),
            type: 'GET',
            success: function(response) {
                console.log(response);
            },
            error: function(error) {
                console.log(error);
            }
        });
    });
});
